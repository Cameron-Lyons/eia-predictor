"""
Taken from https://stackoverflow.com/users/2867928/kasravnd
From their response in: https://stackoverflow.com/questions/31465199/extending-pythons-os-walk-function-on-ftp-server
"""

from os import path as ospath


class FTPWalk:
    """
    This class is contain corresponding functions for traversing the FTP
    servers using BFS algorithm.
    """
    def __init__(self, connection):
        self.connection = connection

    def listdir(self, _path):
        """
        return files and directory names within a path (directory)
        """

        file_list, dirs, nondirs = [], [], []
        try:
            self.connection.cwd(_path)
        except Exception as exp:
            print ("the current path is : ", self.connection.pwd(), exp.__str__(),_path)
            return [], []
        else:
            self.connection.retrlines('LIST', lambda x: file_list.append(x.split()))
            for info in file_list:
                ls_type, name = info[0], info[-1]
                if ls_type.startswith('d'):
                    dirs.append(name)
                else:
                    nondirs.append(name)
            return dirs, nondirs

    def walk(self, path='/'):
        """
        Walk through FTP server's directory tree, based on a BFS algorithm.
        """
        dirs, nondirs = self.listdir(path)
        yield path, dirs, nondirs
        for name in dirs:
            path = ospath.join(path, name)
            yield from self.walk(path)
            # In python2 use:
            # for path, dirs, nondirs in self.walk(path):
            #     yield path, dirs, nondirs
            self.connection.cwd('..')
            path = ospath.dirname(path)