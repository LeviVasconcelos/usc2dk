''' 
0 r ankle, 
1 - r knee, 
2 - r hip, 
3 - l hip, 
4 - l knee, 
5 - l ankle, 
6 - pelvis, 
7 - thorax, 
8 - upper neck, 
9 - head top, 
10 - r wrist, 
11 - r elbow, 
12 - r shoulder, 
13 - l shoulder, 
14 - l elbow, 
15 - l LWrist

3, 2, 1, 6, 7, 8, 0, 13, 14, 15, 27, 26, 25, 17, 18, 19
joint	Humans	Penn	LSP    MPII
Hip	0	                6
RHip	1	8	2       2 
RKnee	2	10	1       1 
RFoot	3	12	0       0 
LHip	6	7	3       3 
LKnee	7	9	4       4 
LFoot	8	11	5       5 
Spine	12	
Thorax	13	                7 
Neck	14		12      8 
Head	15	0	13      9 
LSho	17	1	9       13 
LElbow	18	3	10      14 
LWrist	19	5	11      15  
RSho	25	2	8       12 
RElbow	26	4	7       11 
RWrist	27	6	6       10 
'''
'''
0 : 3
1 : 2
2 : 1
3 : 6
4 : 7
5 : 8
6 : 27
7 : 26
8 : 25
9 : 17
10 : 18
11 : 19
12 : 14
13 : 15
'''
'''
HM to MPII
0 - 3 r ankle, 
1 - 2 r knee, 
2 - 1 r hip, 
3 - 6 l hip, 
4 - 7 l knee, 
5 - 8 l ankle, 
6 - 0 pelvis, 
7 - 13 thorax, 
8 - 14 upper neck, 
9 - 15 head top, 
10 - 27 r wrist, 
11 - 26 r elbow, 
12 - 25 r shoulder, 
13 - 17 l shoulder, 
14 - 18 l elbow, 
15 - 19 l LWrist
'''
'''
MPII to HM_simpli
0 - 6 pelvis, 
1 - 2 r hip, 
2 - 1 r knee, 
3 - 0 r foot, 
4 - 3 l hip, 
5 - 4 l knee, 
6 - 5 l foot, 
7 - 7 thorax, 
8 - 8 upper neck, 
9 - 13 l should, 
10 - 14 l elbow, 
11 - 15 l wrist, 
12 - 12 r shoulder, 
13 - 11 r elbow, 
14 - 10 r wrist, 
'''
'''
MPII to PENN
0 - 9 head
1 - 13 l_shoulder 
2 - 12 r_shoulder
3 - 14 l_elbow
4 - 11 r_elbow
5 - 15 l_wrist
6 - 10 r_wrist
7 - 3 l_hip
8 - 2 r_hip
9 - 4 l_knee
10 - 1 r_knee
12 - 5 l_ankle
13 - 0 r_ankle
'''

import numpy as np
HUMANS_TO_MPII=np.array([3,2,1,6,7,8,0,13,14,15,27,26,25,17,18,19])
HUMANS_TO_PENN=np.array([15,17,25,18,26,19,27,6,1,7,2,8,3])
HUMANS_TO_LSP=np.array([3, 2, 1, 6, 7, 8, 27, 26, 25, 17, 18, 19, 14, 15])
MPII_TO_HUMANS=np.array([6,2,1,0,3,4,5,7,8,13,14,15,12,11,10])
MPII_TO_PENN=np.array([9,13,12,14,11,15,10,3,2,4,1,5,0])
skeletons = {}
skeletons['humans'] = [(0, 1), (1, 2), (2, 3), (0, 4),
                       (4, 5), (5, 6), (0, 7), (7, 8),
                       (7, 9), (9, 10), (10, 11), (7, 12),
                       (12, 13), (13, 14)]

skeletons['penn'] = [(0, 1), (0, 2), (1, 2), (1, 3),
                     (2, 4), (3, 5), (4, 6), (1, 7),
                     (2, 8), (7, 8), (7, 9), (8, 10),
                     (9, 11), (10, 12)]

skeletons['lsp'] = [(12, 13), (12, 8), (12, 9), (8, 7),
                     (7, 6), (9, 10), (10, 11), (8, 2),
                     (9, 3), (2, 1), (1, 0), (3, 4),
                     (4, 5), (2,3), (9,8)]

skeletons['mpii'] = [(0,1), (1,2), (2,6), (6,3), 
                     (3,4), (4,5), (6,7), (7,8), 
                     (8,9), (13, 7), (7, 12), (10,11), 
                     (11,12), (15,14), (14,13)]


