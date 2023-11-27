import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
import cv2

class Propocess_Dataset(tdata.Dataset):
    def __init__(self, folder, norm=True, target="semantic", cache=False, folds=None,
        reference_date=" ", mono_date=None, sats=["S2"],):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and panoptic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
            cache (bool): If True, the loaded samples stay in RAM, default False.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        """
        super(Propocess_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mono_date = (datetime(*map(int, mono_date.split("-"))) if isinstance(mono_date, str) else mono_date)
        self.memory = {}
        self.memory_dates = {}
        self.target = target
        self.sats = sats

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(index=self.meta_patch.index, columns=self.date_range, dtype=int)
            for pid, date_seq in dates.iteritems():
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(lambda x: (datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))- self.reference_date).days)
                date_table.loc[pid, d.values] = 1

            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()}

        print("Done.")

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat([self.meta_patch[self.meta_patch["Fold"] == f] for f in folds])

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(os.path.join(folder, "NORM_{}_patch.json".format(s)), "r") as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None
        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {s: np.load(
                os.path.join(self.folder,"DATA_{}".format(s),"{}_{}.npy".format(s, id_patch),)).astype(np.float32)
                for s in self.sats}  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}

            if self.norm is not None:
                data = {
                    s: (a - self.norm[s][0][None, :, None, None]) / self.norm[s][1][None, :, None, None]
                    for s, a in data.items()}

            if self.target == "semantic":
                target = np.load(os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)))
                target = torch.from_numpy(target[0].astype(int))
            else:
                raise Exception('Do not exit the target')

            if self.cache:
                self.memory[item] = [data, target, target_boundary]

        else:
            data, target, target_boundary = self.memory[item]

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats}
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = data[self.mono_date]
                dates = dates[self.mono_date]
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = (dates - mono_delta).abs().argmin()
                data = data[mono_date]
                dates = dates[mono_date]

        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]

        return (data, dates), target

def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = Propocess_PASTIS_Dataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            print("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0] # [sat]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))