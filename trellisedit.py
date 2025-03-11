import torch
from trellis.pipelines.trellis_edit import train, inference


if __name__ == "__main__":
    '''
    human_xyz = torch.load("overfit/human_coords.pt")
    human_lat = torch.load("overfit/human_feat.pt")
    robot_xyz = torch.load("overfit/robot_coords.pt")
    robot_lat = torch.load("overfit/robot_feat.pt")
    '''
    lr = 5e-5
    n_epoch = 2000#5000
    train_modes = ["baseballcap","bluehat", "redhat", "bow", "glasses"]#["baseballcap","bluehat","bow","bowtie","crown",
    #"glasses","redhat","scarf","sunglasses","tie"]
    test_modes = ["brownhat","greenhat","necklace","tiara"]
    train(lr, n_epoch, train_modes, test_modes)
    #inference(train_modes, test_modes)