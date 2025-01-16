import torch
from trellis.pipelines.trellis_edit import train


if __name__ == "__main__":
    human_xyz = torch.load("overfit/human_coords.pt")
    human_lat = torch.load("overfit/human_feat.pt")
    robot_xyz = torch.load("overfit/robot_coords.pt")
    robot_lat = torch.load("overfit/robot_feat.pt")
    lr = 5e-5
    n_epoch = 5000
    train(lr, n_epoch, human_xyz, human_lat, robot_xyz, robot_lat)