from torch.utils.data import Dataset
import scipy.io as sio

class Gesture(Dataset):
    def __init__(self, train_or_test, transform=None, target_transform=None):
        super(Gesture, self).__init__()
        mat_fname = 'DVS_gesture_100.mat'
        mat_contents = sio.loadmat(mat_fname)
        if train_or_test == 'train':
            self.x_values = mat_contents['train_x_100']
            self.y_values = mat_contents['train_y']
        elif train_or_test == 'test':
            self.x_values = mat_contents['test_x_100']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index, :, :]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)