# easymocap utility functions
import os
import cv2
import numpy as np


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.8f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera(intri_path: str, extri_path: str, cam_names=[]):
    assert os.path.exists(intri_path), intri_path
    assert os.path.exists(extri_path), extri_path

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format(cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Tvec = extri.read('T_{}'.format(cam))

        Rvec = extri.read('R_{}'.format(cam))
        if Rvec is not None:
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read('Rot_{}'.format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['Rvec'] = Rvec
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = - Rvec.T @ Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams


def write_camera(cameras: dict, path: str):
    from os.path import join
    os.makedirs(path, exist_ok=True)
    intri_name = join(path, 'intri.yml')  # TODO: make them arguments
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    cam_names = [key_.split('.')[0] for key_ in cameras.keys()]
    intri.write('names', cam_names, 'list')
    extri.write('names', cam_names, 'list')
    for key_, val in cameras.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        if 'dist' not in val:
            val['dist'] = np.zeros((5, 1))
        intri.write('dist_{}'.format(key), val['dist'])
        if 'R' not in val.keys():
            val['R'] = cv2.Rodrigues(val['Rvec'])[0]
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
