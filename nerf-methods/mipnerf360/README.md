# Mipnerf360 with Depth Supervision


* We have implemented a simple LLFF dataloader to train a depth-supervised neural raidance field using MipNeRF-360.

* Using the script "./scripts/train_kitti.sh" and the default using configuration file is "./config/360.gin"

* Note that the "near" and "far" parameters would be atuomatically adjusted according to the scaling factor computed inside the dataset loader, if you turn on "auto_adjust_near_far". If you don't want it modify these two paramters, please remeber to set "auto_adjust_near_far" to False.

* You need to place your depth data to the folder "depths" or "depth_x", where "x" is the scaling factor you set in your configuration file.

* Enjoy yourself! Contact with Chenming Wu if you need any help.