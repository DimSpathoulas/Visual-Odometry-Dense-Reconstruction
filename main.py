import cv2
import numpy as np
import json



def read_ground_truth_file(path):
    with open(path, "r") as f:
        ground_truth = json.load(f)
    return ground_truth


class VOD:
    def __init__(self, path, lucas_kanade, camera_params, k):
        self.path = path
        self.lucas_kanade = lucas_kanade
        self.camera_params = camera_params
        self.k = k
        self.orb = cv2.ORB_create(self.k)
        self.thresh = 750
        self.num = 0
        self.kps = None
        self.init = 0
        self.trajectory = np.zeros((720, 720, 3), np.uint8)
        self.cords = None
        self.Ee = 0.8


    def read_images_pair(self, i):
        img1_path = f"{self.path}/{str(i).zfill(6)}.png"
        img2_path = f"{self.path}/{str(i + 1).zfill(6)}.png"
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        return img1, img2



    def calc(self, img1, img2):

        if self.num <= self.thresh:
            kps1, _ = self.orb.detectAndCompute(img1, None)
            kp_coords1 = np.array([kp.pt for kp in kps1], dtype=np.float32)

        else:
            kps1 = self.kps
            kp_coords1 = self.cords.copy()


        kps2, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, kp_coords1, None, **self.lucas_kanade)

        kps2_coords = kps2[status.ravel() == 1]

        kps1_filtered = [kp1 for kp1, stat in zip(kps1, status.ravel()) if stat == 1]
        kps1 = np.array(kps1_filtered)

        kps1_coords = np.array([kp.pt for kp in kps1], dtype=np.float32)

        kps2 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kps2_coords]
        self.cords = kps2_coords.copy()
        self.kps = kps2
        self.num = self.cords.shape[0]
        return (kps1,kps1_coords,kps2_coords)



    def flow(self,img1,kps1,kps1_coords,kps2_coords,img2):

        # Draw keypoints on the first image
        img1_kps = cv2.drawKeypoints(img1, kps1, None, color=(0, 0, 255),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Draw optical flow vectors on the second image
        img2_flow = img2.copy()
        for kp1, kp2 in zip(kps1_coords, kps2_coords):
            x1, y1 = kp1
            x2, y2 = kp2
            cv2.arrowedLine(img2_flow, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)



        merged = cv2.vconcat([img1_kps, img2_flow])
        cv2.imshow("Merged Images", merged)

    def draw_trajectory(self, pred_t: list[float], ground_truth_t: list[float]) -> None:
        # ground truth trajectory
        # round numbers
        ground_truth_int = [int(round(t)) for t in ground_truth_t]
        # get center
        center = (ground_truth_int[0] + 400, ground_truth_int[2] + 550)
        # draw points on trajectory image
        self.trajectory = cv2.circle(
            self.trajectory,
            center=center,
            radius=1,
            color=(255, 0, 0),
            thickness=-1
        )

        # prediction trajectory
        # round numbers
        pred_int = [int(round(t)) for t in pred_t]
        # get center
        center = (pred_int[0] + 400, pred_int[2] + 550)
        # draw points on trajectory image
        self.trajectory = cv2.circle(
            self.trajectory,
            center=center,
            radius=1,
            color=(0, 0, 255),
            thickness=-1
        )


    def forward(self):
        gt_data = read_ground_truth_file("ground_truth.json")

        # Main loop
        i = 0
        while True:
            # Read the image pair
            img1, img2 = self.read_images_pair(i)

            kps1, kps1_coords , kps2_coords = self.calc(img1, img2)

            self.flow(img1, kps1, kps1_coords, kps2_coords, img2)
            key = cv2.waitKey(1)

            # Calculate essential matrix
            E, _ = cv2.findEssentialMat(kps1_coords, kps2_coords, cameraMatrix=self.camera_params,
                                        method=cv2.RANSAC, prob=0.99999, threshold=self.Ee)

            # Recover pose
            retval, R, t, mask = cv2.recoverPose(E, kps1_coords, kps2_coords, cameraMatrix=self.camera_params)


            t_prior = np.zeros_like(t)
            t_post = np.zeros_like(t)
            t_prior = np.array([gt_data[i]]).T
            t_post = np.array([gt_data[i + 1]]).T

            if self.init == 0:
                self.init = 1
                scale = np.zeros((1,1))
                t_old = t_prior
                t_inter = np.zeros_like(t)
                MSE = 0.0
                R_prior = np.eye(*R.shape)  # identity matrix for start
                R_post = np.zeros_like(R)
                t_real = np.zeros_like(t)


            scale = np.array([[(np.linalg.norm(t_post - t_prior)) ** 2]])

            if scale > 0.007:
                t_inter = t_old + (scale @ (t.T @ R_prior)).T
                R_post = R @ R_prior

                H = np.hstack((R_post, t_inter))
                H = np.vstack((H, np.array([0, 0, 0, 1])))
                t_real = np.vstack((t_prior, [1]))
                t_real = H @ t_real
                t_real = t_real[:3]
                t_old = t_real - (t_prior + t_inter)
                R_prior = np.transpose(R) @ np.transpose(R_post)

            errors = t_post[:, 0] - t_real[:, 0]
            squared_errors = np.square(errors)
            MSE = np.mean(squared_errors)
            print("error is: MSE = ", MSE if MSE > 1e-3 else 0.0)


            self.draw_trajectory(t_real.flatten().tolist(), t_post.flatten().tolist())
            cv2.imshow("trajectory", self.trajectory)

            i = i + 1

            if key == ord('q'):
                break





path = "imgs"
lucas_kanade_params = {
    "winSize": (21, 21),
    "maxLevel": 5
}

k = 3000
camera_params = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]])

vod = VOD(path, lucas_kanade_params, camera_params, k)
vod.forward()
