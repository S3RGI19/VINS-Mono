#!/usr/bin/env python
import rosbag
import numpy as np
import tf 
import csv

# Replace with your bag file name
bag_file = 'odometry_record.bag'

# Containers for messages from each topic
odom_msgs = []
cam_pose_msgs = []
extrinsics_msgs = []
cam_pose_visual_msgs = []

# Read the bag and store messages from the two topics
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/vins_estimator/odometry',
                                                    '/vins_estimator/camera_pose',
                                                    '/vins_estimator/extrinsic',
                                                    '/vins_estimator/camera_pose_visual']):
        if topic == '/vins_estimator/odometry':
            odom_msgs.append(msg)
        elif topic == '/vins_estimator/camera_pose':
            cam_pose_msgs.append(msg)
        elif topic == '/vins_estimator/extrinsic':
            extrinsics_msgs.append(msg)
        elif topic == '/vins_estimator/camera_pose_visual':
            cam_pose_visual_msgs.append(msg)

print("Number of odometry messages: {}".format(len(odom_msgs)))
print("Number of camera_pose messages: {}".format(len(cam_pose_msgs)))
print("Number of extrinsic messages: {}".format(len(extrinsics_msgs)))
print("Number of camera_pose_visual messages: {}".format(len(cam_pose_visual_msgs)))

# # Process the messages pairwise
num = min(len(odom_msgs), len(cam_pose_msgs))
# print("Processing {} message pairs...".format(num))
timestamps = []
T_odoms = []
for i in range(num):
    # Get the i-th messages from each topic
    odom = odom_msgs[i]            # Body pose from pubOdometry: wTb
    cam  = cam_pose_msgs[i]         # Camera pose from pubCameraPose: wTc
    Tbc_msg = extrinsics_msgs[i]     # Extrinsics: bTc
    # Note: cam_pose_visual_msgs is not used in this comparison

    # Build the transformation matrix for odometry (wTb)
    T_odom = np.eye(4)
    odom_pos = odom.pose.pose.position
    T_odom[0:3, 3] = np.array([odom_pos.x, odom_pos.y, odom_pos.z])
    # For orientation, convert quaternion to rotation matrix
    odom_ori = odom.pose.pose.orientation
    q_odom = [odom_ori.x, odom_ori.y, odom_ori.z, odom_ori.w]
    T_odom[0:3, 0:3] = tf.transformations.quaternion_matrix(q_odom)[0:3, 0:3]
    T_odoms.append(T_odom)
    print(f"Message pair {i}: wTb: {q_odom} {T_odom[0:3, 3]}")
    # Build the transformation matrix for the published camera pose (wTc)
    T_cam = np.eye(4)
    cam_pos = cam.pose.pose.position
    T_cam[0:3, 3] = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
    cam_ori = cam.pose.pose.orientation
    q_cam = [cam_ori.x, cam_ori.y, cam_ori.z, cam_ori.w]
    T_cam[0:3, 0:3] = tf.transformations.quaternion_matrix(q_cam)[0:3, 0:3]

    # Build the transformation matrix for the extrinsics (bTc)
    T_bc = np.eye(4)
    # Extract translation from the extrinsics message
    trans = Tbc_msg.pose.pose.position
    T_bc[0:3, 3] = np.array([trans.x, trans.y, trans.z])
    # Extract rotation from the extrinsics message and convert to a rotation matrix
    rot = Tbc_msg.pose.pose.orientation
    q_bc = [rot.x, rot.y, rot.z, rot.w]
    T_bc[0:3, 0:3] = tf.transformations.quaternion_matrix(q_bc)[0:3, 0:3]
    # 
    timestamps.append(odom.header.stamp.to_nsec())
    print("time: {}".format(odom.header.stamp.to_nsec()))
    print("time: {}".format(cam.header.stamp.to_nsec()))
    print("time: {}".format(Tbc_msg.header.stamp.to_nsec()))
    # Now compute the expected camera pose from odometry and extrinsics:
    # wTc_computed = wTb * bTc
    T_computed = T_odom @ T_bc

    # print("Message pair {}: Computed wTc_: \n{}".format(i, T_computed))
    # print("Message pair {}: Published wTc: \n{}".format(i, T_cam))
    # print("Message pair {}: Extrinsics bTc: \n{}".format(i, T_bc))


    # Compare inv(T_computed) with the published camera pose T_cam
    T_diff = np.linalg.inv(T_computed) @ T_cam
    error = np.linalg.norm(T_diff - np.eye(4))
    # print("Message pair {}: Norm difference between computed wTc and published wTc: {:.6f}".format(i, error))
    # print("------------------------------------------------------")
    if i == 2:
        break

# 
gt_pose_path = '/datasets/euroc/raw/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv'
gt_timestamps = []  # in nseconds
gt_poses = [] 
tolerance = 1e6
# load ground truth pose
with open(gt_pose_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # Skip header or comment lines (if any)
        if row[0].startswith('#'):
            continue
        # Parse the row (assuming comma-separated values)
        t = float(row[0])  # Convert to nanoseconds
        px = float(row[1])
        py = float(row[2])
        pz = float(row[3])
        qw = float(row[4])
        qx = float(row[5])
        qy = float(row[6])
        qz = float(row[7])
        for odom_time in timestamps:
            if abs(t - odom_time) < tolerance:
                print(t)
                gt_timestamps.append(t)
                T_gt = np.eye(4)
                T_gt[0:3, 3] = np.array([px, py, pz])
                T_gt[0:3, 0:3] = tf.transformations.quaternion_matrix([qx, qy, qz, qw])[0:3, 0:3]
                gt_poses.append(T_gt)
                print(f"gt_pose: {qx}, {qy}, {qz}, {qw}, {px}, {py}, {pz}")
                # break

# print the prior based on vins
T0 = T_odoms[0]
for i in range(0, len(T_odoms)):
    Tvins =  np.linalg.inv(T0) @ T_odoms[i]
    print("timestamps: {}, T_vins\n: {}".format(timestamps[i], T_odoms[i]))
    # print the prior based on vins
    Tgt = T0 @ np.linalg.inv(gt_poses[0]) @ gt_poses[i]
    print("timestamps: {}, T_gt\n: {}".format(gt_timestamps[i], Tgt))
    # print the prior based on gt
    # print("T0: {}".format(T0))

