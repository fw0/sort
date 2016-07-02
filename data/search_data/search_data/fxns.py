import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import python_utils.python_utils.sklearn_utils as sklearn_utils
import constants
import os
import subprocess
import string
import pdb
import tempfile
import sys
import datetime
import numpy as np

from multiprocessing import Process, Value, Lock
lock = Lock()

def compose(*args):
    def h(x):
        ans = x
        for f in args:
            ans = f(ans)
        return ans
    return h

class file_path(object):

    def __init__(self, path, temp=False):
        self.path = path
        self.temp = temp

    @property
    def name(self):
        return os.path.split(self.path)[-1]

    def get_reader(self):
        return open(self.path, 'r')

    def get_writer(self):
        return open(self.path, 'wb')

    def copy_from(self, from_file_path):
        reader = from_file_path.get_reader()
        writer = self.get_writer()
        for line in reader:
            writer.write(line)
        writer.close()

    def __del__(self):
        print 'deleting', self.path, self.temp
        sys.stdout.flush()
        if self.temp:
            print 'removing', self.path
            sys.stdout.flush()
            try:
                os.remove(self.path)
            except OSError:
                print 'no file to remove', self.path
                sys.stdout.flush()

class file_collection_path(object):

    def __init__(self, file_paths):
        self.file_paths = file_paths #[file_path(path) for path in raw_file_paths]

    def __iter__(self):
        return iter(self.file_paths)

class folder_path(file_collection_path):

    def copy_from(self, from_file_collection_path):
        assert not (self.folder_path is None)
        folder_getter(self.folder_path)
        self.file_paths = []
        for (i, from_file_path) in enumerate(from_file_collection_path.file_paths):
            read_file_name = file_path.name
            write_file_path = file_path('%s/%s' % (self.folder_path, from_file_path.name))
            write_file_path.copy_from(from_file_path)
            self.file_paths.append(write_file_path)

    def __init__(self, _folder_path):
        self.folder_path = _folder_path
        #assert os.path.exists(_folder_path)
        folder_getter(self.folder_path, False)
        self.file_paths = [file_path('%s/%s' % (_folder_path, path)) for path in os.listdir(_folder_path)]

def apply_to_file_collection_path(mapper, f, _file_collection_path):
    """
    f takes in file_path, outputs file_path.  use it to make folder_path
    """
    def run(read_file_path):

        def get_temp_path():
            return '%s_%s' % (read_file_path.path, str(datetime.datetime.now()))

        temp_path = get_temp_path()
        print temp_path
        sys.stdout.flush()
        
        write_file_path = file_path(temp_path, temp=True)
        f(read_file_path, write_file_path)
        return write_file_path

    write_file_paths = mapper(run, list(_file_collection_path))
    return file_collection_path(write_file_paths)

def folder_getter(path, override=True):
    if override:
        try:
            import shutil
            shutil.rmtree(path)
        except Exception as e:
            pass
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def split_data_file(path, num_lines, path_to_folder = lambda s: '%s_%s' % (s, 'split'), folder_getter = folder_getter):

    # create output folder
    _folder_path = path_to_folder(path)
    _folder_path = folder_getter(path = _folder_path)
    
    # create temp of original file without first line to split
    no_head_path = '%s.%s' % (path,'no_header_temp')
    if os.path.exists(no_head_path):
        os.remove(no_head_path)
    no_head_cmd = 'tail -n +2 %s > %s' % (path, no_head_path)
    subprocess.call(no_head_cmd, shell=True)

    # split the files
    split_cmd = 'split -d -l %d -a 4 %s %s/no_head_' % (num_lines, no_head_path, _folder_path)
    subprocess.call(split_cmd, shell=True)

    # create split files with header
    with_head_split_paths = []
    for no_head_split_file in os.listdir(_folder_path):
        no_head_split_path = '%s/%s' % (_folder_path, no_head_split_file)
        with_head_split_path = '%s/%s' % (_folder_path, string.split(no_head_split_file, sep='_')[-1])
        add_head_cmd = 'head -n 1 %s > %s' % (path, with_head_split_path)
        subprocess.call(add_head_cmd, shell=True)
        add_content_cmd = 'cat %s >> %s' % (no_head_split_path, with_head_split_path)
        with_head_split_paths.append(with_head_split_path)
        subprocess.call(add_content_cmd, shell=True)
        rm_split_no_head_cmd = 'rm %s' % no_head_split_path
        subprocess.call(rm_split_no_head_cmd, shell=True)

    rm_no_head_cmd = 'rm %s' % no_head_path
    subprocess.call(rm_no_head_cmd, shell=True)

    return folder_path(_folder_path)

def df_f_to_path_f(f):

    def h(read_file_path, write_file_path):
        df = pd.read_csv(read_file_path.path, sep='\t', index_col=0)
        ans = f(df)
        ans.to_csv(write_file_path.path, sep='\t')

    return h

def categorical_df_to_onehot_df(df, level_to_idxs):
    """
    ok for level_to_idxs to have too much information
    """

    verts = []

    for feat in df.columns:
        

        def onehot(l, k):
            x = np.zeros(l-1, dtype=int)
            if k != l-1:
                x[k] = 1
            return x

        def col_to_df(feat, col, level_to_idx):
            """

            """
            num_levels = len(level_to_idx)
            rows = []
            for (index, level) in col.iteritems():
                rows.append(onehot(len(level_to_idx), level_to_idx[level]))
            idx_to_level = [x for x in xrange(num_levels)]
            for (level, idx) in level_to_idx.iteritems():
                if idx < num_levels - 1:
                    idx_to_level[idx] = level

            try:
                ans = pd.DataFrame.from_records(rows, columns=['%s=%s' % (str(feat),str(level)) for level in idx_to_level[:-1]], index=col.index)
            except:
                pdb.set_trace()
            return ans

        verts.append(col_to_df(feat, df[feat], level_to_idxs[feat]))

    return pd.concat(verts, axis=1)
        

def a_data():
    return pd.read_csv('%s/%s' % (constants.data_folder, 'data.2016-04-01.txt'),sep='\t')




class file_handle_deprecated(object):

    @classmethod
    def open(cls, *args, **kwargs):
        return cls(open(*args, **kwargs))

    def __init__(self, f):
        self.f = f

    def write(self, *args, **kwargs):
        return self.f.write(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.f.seek(*args, **kwargs)

    def copy_from(self, read_file_handle):
        read_file_handle.seek(0)
        self.seek(0)
        for line in read_file_handle:
            #print line
            #import sys
            #sys.stdout.flush()
            self.write(line)
