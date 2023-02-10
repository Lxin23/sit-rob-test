import sys

import xacro
from urdf_parser_py.urdf import *
from joint import *

from robot_links import *
import gazebo_plugin

sit_basketball = Robot('sit_basketball_robot')

# 创建footprint以及base_link
footprint_link = create_footprint_link(sit_basketball)
base_link = create_base_link(sit_basketball, footprint_link)

# 创建激光雷达
laser_link = create_laser_link(sit_basketball, base_link)

# 创建IMU
imu_link = create_imu_link(sit_basketball, base_link)

# 创建左右驱动轮
left_drive_wheel_link, left_drive_wheel_joint = create_drive_wheel(sit_basketball, base_link, 'left')
right_drive_wheel_link, right_drive_wheel_joint = create_drive_wheel(sit_basketball, base_link, 'right')

# 创建支撑轮
front_support_wheel_link = create_support_wheel(sit_basketball, base_link, 'front')
# back_support_wheel_link = create_support_wheel(sit_basketball,base_link,'back')

# # 创建底盘与电池底盘的连接杆
# battery2base_link = create_metal_link(sit_basketball,
#                                       base_link,
#                                       'battery2base',
#                                       battery2base_distance)

# # 创建电池底盘
# battery_base_link = create_battery_base_link(sit_basketball, battery2base_link)

# # 创建左右电池
# create_battery_link(sit_basketball, battery_base_link, 'left')
# create_battery_link(sit_basketball, battery_base_link, 'right')

# # 创建电池底盘与机械臂底盘的连接杆
# arm2battery_link = create_metal_link(sit_basketball,
#                                      battery_base_link,
#                                      'arm2battery',
#                                      arm2battery_distance)

# # 创建机械臂底盘
# arm_base_link = create_arm_base_link(sit_basketball, arm2battery_link)

# # 创建机械臂底盘与深度相机底盘的连接杆
# camera2arm_link = create_metal_link(sit_basketball,
#                                     parent_link=arm_base_link,
#                                     name_prefix='camera2arm',
#                                     length=camera2arm_distance)

# # 创建深度相机底盘
# camera_base_link = create_camera_base_link(sit_basketball, camera2arm_link)

# # 添加深度相机
# camera_link = add_d435i_rgbd_camera_link(sit_basketball, camera_base_link)

# 创建机械臂
# add_robot_arm(sit_service,arm_base_link)

# 添加差速控制器
gazebo_plugin.add_drive_controller(sit_basketball,
                                   left_drive_wheel_joint,
                                   right_drive_wheel_joint,
                                   footprint_link.name)

# 添加激光雷达插件
# gazebo_plugin.add_laser_sensor(sit_basketball, laser_link)

# 添加IMU插件
# gazebo_plugin.add_imu_sensor(sit_service, imu_link)

# # 添加深度相机插件
# gazebo_plugin.add_rgbd_camera_sensor(sit_service, camera_link)

print(sit_basketball)
