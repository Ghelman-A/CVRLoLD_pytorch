import pandas as pd
import random
import numpy as np
import os

from pathlib import Path
import pickle
from copy import deepcopy
from itertools import combinations


class PrepareDataLists:
    """
        This class prepares the labeled data for linear evaluation, semi-supervised, and supervised
        cases.
        The evaluation and test datasets are prepared using the 10 crop, 3 view in the paper and are
        the same for all of the above 3 approaches.

    NOTE:
        Only in the semi_train_eval method random function is used which is also controlled by assiging
        a random.seed beforehand. As a result, so long as the input excel files are the same and then
        percentage of the semi-supervised case is the same, the resulting pickle files are not different.
        Meaning that the results of training using them can be compared with previous cases.
    """
    def __init__(self, config):
        self.config = config
        self.prep_config = config['data_prep']
        self.classes = config['OBJECT_ACTION_NAMES']['Excavator']

        self.train_csv = config['data_prep']['train_csv_dir']
        self.eval_csv = config['data_prep']['eval_csv_dir']
        self.test_csv = config['data_prep']['test_csv_dir']

        self.save_dir = config['ds_list_dir']
        with open(config['ds_hdf5_idx_dir'], 'rb') as f:
            self.dset_idx = pickle.load(f)

    def ssl_train(self):
        """
        This method extracts the cyclic clips from each video for every frame and returns the
        result as a list of clips per videos.

            output format per row in the dataframe: [list of clips]
        :return:
        """
        clip_len = self.config['SSL']['clip_len']
        skip_step = self.config['SSL']['skip_step']
        overlap = self.config['SSL']['overlap']

        train_df = pd.read_excel(self.train_csv, index_col=0)

        self.__prep_ssl_data__(clip_len, skip_step, overlap, train_df, 'SSL')

    def sup_train(self):
        """
            This class creates DataFrame for supervised training dataset along with its corresponding save file name,
            and passes the result as arguments to the __prep_train_data__ method for final dataset creation and saving.
        :return: None
        """
        train_df = pd.read_excel(self.train_csv, index_col=0)
        mode = 'supervised'
        self.__prep_train_data__(self.config[mode]['clip_len'], self.config[mode]['skip_step'],
                                 self.config[mode]['overlap'], train_df, mode)

    def lin_eval(self):
        """
            This class creates DataFrame for linear evaluation dataset along with its corresponding save file names, 
            and passes the result as arguments to the __prep_train_data__ method for final dataset creation and saving.

            It should be noted that in this case all of the eval dataset is used to train the added linear head (added to the
            output of the SSL training), and there are no eval datasets in this case!
        :return: None
        """
        train_df = pd.read_excel(self.eval_csv, index_col=0)
        mode = 'linear_eval'
        name_pref = 'lin_eval'
        self.__prep_train_data__(self.config[mode]['clip_len'], self.config[mode]['skip_step'],
                                 self.config[mode]['overlap'], train_df, name_pref)

    def semi_train_eval(self):
        """
            This class creates DataFrame for train and evaluation datasets for semi-supervised data
            along with their corresponding save file names, and passes the results as arguments to the 
            __prep_train_data__ method for final dataset creation and saving.

            For now for the 10% case all of the data is used for training and there is no eval set, but for
            the 1% case, the remaining 9% are used for evaluation.
        :return: None
        """
        train_df = pd.read_excel(self.train_csv, index_col=0)
        eval_df = pd.read_excel(self.eval_csv, index_col=0)
        test_df = pd.read_excel(self.test_csv, index_col=0)
        new_df = pd.concat([train_df, eval_df], ignore_index=True)

        #-------------------------------------------------------------------------------------#
        # The reason for including the test size is because the percent of the semi dataset   #
        # is relative to the total dataset size and not only the sum of train and eval.       #
        #-------------------------------------------------------------------------------------#
        total_ds_size = len(new_df) + len(test_df)
        new_eval_size = np.floor(total_ds_size * (self.config['semi']['semi_percent'] / 100.0))
        per_class_size = np.ceil(new_eval_size / 3).astype(np.int)

        ##### Getting a list of data in each activity class
        dig_idx = []
        load_idx = []
        swing_idx = []

        vid_name_header = self.prep_config['raw_dataset_csv_headers'][0]
        for idx, data in new_df.iterrows():
            if 'digging' in data[vid_name_header].lower():
                dig_idx.append(idx)
            elif 'loading' in data[vid_name_header].lower():
                load_idx.append(idx)
            elif 'swinging' in data[vid_name_header].lower():
                swing_idx.append(idx)

        #-------------------------------------------------------------------------------------#
        #      Preparing the 1% and 10% datasets for the semi-supervised training case        #
        #-------------------------------------------------------------------------------------#
        random.Random(100).shuffle(dig_idx)         # For reproducibility
        random.Random(100).shuffle(load_idx)        # For reproducibility
        random.Random(100).shuffle(swing_idx)       # For reproducibility
        
        semi_idx = dig_idx[:per_class_size] + load_idx[:per_class_size] + swing_idx[:per_class_size]
        ssl_idx = dig_idx[per_class_size:] + load_idx[per_class_size:] + swing_idx[per_class_size:]

        semi_df = deepcopy(new_df.iloc[semi_idx, :])
        ssl_df = deepcopy(new_df.iloc[ssl_idx, :])

        #-----   Creating and saving the datasets
        #-----   Changed the SSL data perparation config to that of the SSL mode, now the semi mode data prep
        #-----   config is only used for creating the supervised portion of the dataset. This allows for more
        #-----   freedom in preparing the data for the semi case and ensures more compatibility with the SSL mode.
        mode = 'semi'
        name_pref = f'semi_{self.config["semi"]["semi_percent"]}'
        self.__prep_train_data__(self.config[mode]['clip_len'], self.config[mode]['skip_step'],
                                 self.config[mode]['overlap'], semi_df, name_pref)
        self.__prep_ssl_data__(self.config['SSL']['clip_len'], self.config['SSL']['skip_step'],
                               self.config['SSL']['overlap'], ssl_df, name_pref)

    def test_n_sup_eval(self):

        full_dir = [os.path.join(self.save_dir, case) for case in ['supervised', 'test']]
        for dir in full_dir:
            os.makedirs(dir, exist_ok=True)
        
        file_name = f'clip_len_{self.config["supervised"]["eval_clip_len"]}_skip_{self.config["supervised"]["eval_skip"]}' \
                    f'_overlap_{self.config["supervised"]["eval_overlap"]}.pkl'

        self.__test_crop__(self.eval_csv, os.path.join(full_dir[0], 'eval_' + file_name))
        self.__test_crop__(self.test_csv, os.path.join(full_dir[1], file_name))
    
    def __prep_ssl_data__(self, clip_len, skip_step, overlap, train_df, name_prefix):
        """
        This method extracts the clips from each video in a cyclic manner according to the recommendation in the
        Ha et al. paper. All of the extracted clips are treated as separate data samples on which the model can
        be trained.

            ####### output format in each row of the dataframe: [idx, clip]
            The idx is from the index of the selected frame in the h5py dataset. In other words, this function only
            selects the index of the train data or the index of the frames in each clip, and the final data are queried
            from the h5py file during training in the Dataset class. (i.e., the get_item() function)

        :param clip_len:
        :param skip_step:
        :param overlap:
        :param train_df: The input dataframe with information about the dataset, used to extract clips.
        :param name_prefix: A way to handle different cases of SSL and semi-supervised.
        :return: None
        """
        save_dir = os.path.join(self.save_dir, f'{name_prefix}')
        os.makedirs(save_dir, exist_ok=True)
        file_name = f'clip_len_{clip_len}_skip_{skip_step}_overlap_{overlap}.pkl'
        file_name = "SSL_" + file_name if 'semi' in name_prefix else file_name
        save_file = os.path.join(save_dir, file_name)

        final_frames = []
        headers = self.prep_config['raw_dataset_csv_headers']
        overlap_step = clip_len - overlap  # Reason: if overlap = 9 & clip = 16 --> overlap_step = 7
        assert overlap_step != 0, "Overlap size is the same as clip length in SSL"

        for _, data in train_df.iterrows():
            vid_dir = Path(data[headers[1]])

            vid_clips = []
            vid_idx = []
            frames = sorted(list(vid_dir.glob("*.PNG")))[::skip_step]

            for frame_idx in range(0, len(frames), overlap_step):   # range(start, end, step)
                clip = []
                start_idx = frame_idx

                ##### Cyclic Loop over the frame list to get the required clip length
                while len(clip) < clip_len:

                    start_idx %= len(frames)
                    clip.append(frames[start_idx])
                    start_idx += 1

                vid_clips.append(clip)
                vid_idx.append([self.dset_idx[str(clip_)] for clip_ in clip])

            final_frames.append((vid_idx, vid_clips))

        if self.config['SSL']['all_ssl_comb']:
            final_frames, min_len = self.__get_all_combinations__(final_frames)
            save_file = save_file[:-4] + f'_full_comb_min_len_{min_len}.pkl'

        data_df = pd.DataFrame(final_frames, columns=['data', 'frame_names'])
        print(f"Length of {save_file}: {len(data_df)}")

        with open(save_file, 'wb') as f:
            pickle.dump(data_df, f)
    
    def __prep_train_data__(self, clip_len, skip_step, overlap, train_df, name_prefix):
        """
        This method extracts the clips from each video in a cyclic manner according to the recommendation in the
        Ha et al. paper. All of the extracted clips are treated as separate data samples on which the model can
        be trained.

            ####### output format in each row of the dataframe: [idx, label, clip]
            The idx is from the index of the selected frame in the h5py dataset. In other words, this function only
            selects the index of the train data or the index of the frames in each clip, and the final data are queried
            from the h5py file during training in the Dataset class. (i.e., the get_item() function)

        :param clip_len:
        :param skip_step:
        :param train_df: The input dataframe with information about the dataset, used to extract clips.
        :param name_prefix: A way to handle different cases of supervised, linear evaluation, and semi-supervised.
        :return:
        """

        save_dir = os.path.join(self.save_dir, f'{name_prefix}')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'clip_len_{clip_len}_skip_{skip_step}_overlap_{overlap}.pkl')

        headers = self.prep_config['raw_dataset_csv_headers']
        overlap_step = clip_len - overlap   # Reason: if overlap = 9 & clip = 16 --> overlap_step = 7
        assert overlap_step != 0, f"Overlap size is the same as clip length in {name_prefix}"

        ######################################################################################
        # Creates clips from the input video based on the given skip step and clip len, and  #
        # frame overlap and saves the name of the selected frames in a data frame for use    #
        # during training.                                                                   #
        ######################################################################################
        final_frames = []

        for _, data in train_df.iterrows():
            """
                To simplify clip extraction from input videos based on the skip step and the overlap between 
                clips, this process is broken into two steps. First based on the skip step, the frames are divided
                into non-overlapping groups. For example if skip step is 3 we have:
                    - 0, 3, 6, 9, 12, ...
                    - 1, 4, 7, 10, 13, ...
                    - 2, 5, 8, 11, 14, ...
                Then from each group the frames are extracted with the considered overlap. For example for the
                overlap of 8 in 16 frame clips we will have:
                    - [0, 3, ..., 45], [24, 27, ..., 69]
                    - [1, 4, ..., 46], [25, 28, ..., 70]
                    - [2, 5, ..., 47], [26, 29, ..., 71]
                As can be seen above, even with this method, some clips are only one frame apart. The difference
                with the previous approach is that in that case only one frame would be different between consecutive
                clips but in this case it's like the clips are shifted. One probably good idea is to only use the
                clips of the first row, which is what I have implemented so far!
            """
            curr = Path(data[headers[1]])
            label = self.classes.index(data[headers[0]].split("_")[0].lower())
            frame_list = sorted(list(curr.glob("*.PNG")))[::skip_step]

            for frame_idx in range(0, len(frame_list), overlap_step):   # range(start, end, step)
                clip = []
                start_idx = frame_idx

                ##### Cyclic Loop over the frame list to get the required clip length
                while len(clip) < clip_len:

                    start_idx %= len(frame_list)
                    clip.append(frame_list[start_idx])
                    start_idx += 1

                dset_idx = [self.dset_idx[str(clip_)] for clip_ in clip]
                final_frames.append((dset_idx, label, clip))

        data_df = pd.DataFrame(final_frames, columns=['data', 'label', 'frame_names'])
        print(f"Length of {save_file}: {len(data_df)}")

        ##### Writing the dataset into a pickle file
        with open(save_file, 'wb') as f:
            pickle.dump(data_df, f)
    
    def __test_crop__(self, dataset_csv, save_name):
        """
            This function implements sliding window approach to extract clips from test videos.

            The output is in the format of ([list of clips], label)
        :param dataset_csv: The input dataset (either eval or test)
        :return: None
        """
        ##### Reading the csv dataset
        dataset_df = pd.read_excel(dataset_csv, index_col=0)

        prep_data = []
        headers = self.config['data_prep']['supervised_data_headers']

        ##### Extracting clips and labels. A single activity label is assigned to all clips from a single video
        """
            More detailed description can be found in the __prep_train_data__ method, which implements a similar
            functionality.
        """
        slide_step = self.config["supervised"]["eval_clip_len"] - self.config["supervised"]["eval_overlap"]
        assert slide_step != 0, "Window overlap size is the same as eval clip length!"

        for _, data in dataset_df.iterrows():

            label = self.classes.index(data[headers[0]].split("_")[0].lower())
            vid_path = Path(data[headers[1]])
            frames = sorted(list(vid_path.glob("*.PNG")))[::self.config["supervised"]["eval_skip"]]

            clips = []
            dset_idx = []

            for i in range(0, len(frames), slide_step):
                start_idx = i
                clip = []

                ##### Cyclic Loop over the frame list to get the required clip length
                while len(clip) < self.config["supervised"]["eval_clip_len"]:
                    start_idx %= len(frames)
                    clip.append(frames[start_idx])
                    start_idx += 1

                clips.append(clip)
                dset_idx.append([self.dset_idx[str(clip_)] for clip_ in clip])

            prep_data.append((dset_idx, label, clips))

        data_df = pd.DataFrame(prep_data, columns=['data', 'label', 'frame_names'])
        with open(save_name, 'wb') as f:
            pickle.dump(data_df, f)

    def __get_all_combinations__(self, vid_clip_list) -> list:
        """
            This script extracts all of the possible dual combinations of clips from each video
            The idea is to replace this method with the random selection of temporal clips at runtime
            to see if this will help improving the results.

        Args:
            vid_clip_list (List): A list of all of the clips extracted from all videos
        """
        new_final_frames = []
        min_len = -1                # Keeping track of the min number of clips per video in the dataset
        
        for vid in vid_clip_list:
            
            clip_idx = range(len(vid[0]))
            comb = combinations(clip_idx, 2)
            min_len = len(comb) if len(comb) > min_len else min_len
            
            """
                The format of the data in the vid is (vid_idx, vid_clips) which has all of the clips extracted from a single
                video. The format of the new output list is:
                
                    ((vid_idx for clip1, vid_idx for clip2), (vid_clips for clip1, vid_clips for clip2))
            """
            new_final_frames.append([((vid[0][comb_[0]], vid[0][comb_[1]]), (vid[1][comb_[0]], vid[1][comb_[1]])) for comb_ in comb])
        
        return new_final_frames, min_len
