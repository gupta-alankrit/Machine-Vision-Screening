# Workspace-B individual-component raw videos -- MANIFEST (files not in repo)

Single-component clips used as the source for the **SAM-2 mask bank**
(script `02_build_sam2_mask_bank.py`). Each clip shows one component
(screw / nut / gear variant) rotating on a turntable against a plain
white-wall background so SAM-2 can extract clean per-frame instance
masks. The resulting mask bank is the *paste source* for the copy-paste
augmentation pipeline in script `05_copy_paste_augment.py`.

The files themselves are too large for GitHub and are tracked **out of band**.

## Where to save after obtaining

After receiving the videos from the repo owner (see "How to obtain"
below), place them at this exact path inside your local clone of the
repo so the scripts in `scripts/` can find them without code changes:

```
<repo_root>/data/workspace_B/individual_components/raw_videos/
└── <component_id>_<take>.MOV     (40 files, ~1.4 GB total)
```

## Source location on capture machine

`/home/agupta3129/machine_vision/nikonD5300_recordings/raw_videos/`

## Camera and capture settings

| Property            | Value                                                  |
|---------------------|--------------------------------------------------------|
| Camera              | Nikon D5300 DSLR                                       |
| Sensor              | 24.2 MP APS-C CMOS (used in *photo* mode only)         |
| Video container     | QuickTime (`.MOV`)                                     |
| Video codec         | H.264 / AVC                                            |
| Resolution          | 1280 x 720 (720p)                                      |
| Frame rate          | 59.94 fps                                              |
| Typical clip length | ~30 s (~1,770 frames per clip)                         |

Capture protocol:
- Each component variant is placed on a manual turntable and rotated
  through ~360 deg over the clip so SAM-2 sees every stable resting
  pose.
- Background is a uniform white wall to give SAM-2 a high-contrast
  silhouette with minimal prompting in script
  `02_build_sam2_mask_bank.py`.
- One component per clip, no occlusions.

## Files

40 `.MOV` clips, total ~1.4 GB. Naming convention: `<component_id>_<take_number>.MOV`.

| Filename            | Size (bytes) | SHA256                                                              |
|---------------------|--------------|---------------------------------------------------------------------|
| gear_a_01.MOV       | 36,868,063   | b759c54cbbfee4f46f41ee59a0055af5f9fc3a9c2a536175747a4529660020e6    |
| gear_a_02.MOV       | 36,226,930   | 92b6cc16686381126fecbf1b78ee7a561cfa96d8ec69645348376d021b79657a    |
| gear_b_01.MOV       | 37,482,394   | 9a0ce0877e9e77d1977f338390e4f2af94287e7e051d8d849d10132cd971ae13    |
| gear_b_02.MOV       | 36,365,766   | 3070f0279f86044578d3a2aa5ec88a2fcae9868df805757683e7b5c76f5c2ae6    |
| nut_a_01.MOV        | 36,883,682   | 740dc858c1aae36f6694f76b47b45117b567aa0e0cd34b77414aef0ac75a57f2    |
| nut_a_02.MOV        | 36,915,397   | b9a37fdb47d54dc237abef2ec19082ad6f0255b7a7f19360d6f5b3d6b047e09f    |
| nut_b_01.MOV        | 37,466,399   | d9c13e1a6f07947a61193046f4cdf11198f218d4d12e01d7fac1a305b36c440c    |
| nut_b_02.MOV        | 38,648,649   | 13c042c234ab20c56f0be433023da9d695497a160407ef6efea267d143f5ab36    |
| nut_b_03.MOV        | 37,820,352   | 6a919b9e74ae9442446df01aa9acff0da6c4a9daa5f6473590756a59d7e31ab5    |
| nut_c_01.MOV        | 37,692,359   | ed6b5bc7c998ec6eb71628a228b28615be56e9a2f847ce0c887b50d8aaa6a487    |
| nut_c_02.MOV        | 37,988,548   | 8ac19003f4ed08aad4f22b552b8bc12c25e2bd9d5469778a6abde4aab9970a54    |
| nut_e_01.MOV        | 36,739,607   | 9b67932c63b8093bb60350638c726acccf7845cb01c69ac71359e859967f4766    |
| nut_e_02.MOV        | 36,770,247   | 0b0e10a215bd305b38142e997f832320dd2050bf4a09cb24af72bcb39162206e    |
| nut_f_01.MOV        | 37,799,305   | 1aa4957dec0d545319527e452f0e4bae5e269381a0bf1cfd3d10bc968b04b121    |
| nut_f_02.MOV        | 36,832,260   | bdb28241a4e53fff66831a707856712f178b570a57f796447befd4aeece0d83d    |
| nut_f_03.MOV        | 37,515,691   | e05d6618a7d7a76307bcd601bdf81890205829079665ecf2eba30c105eb77a3e    |
| nut_g_01.MOV        | 36,565,058   | 5076c1324fdc4e25c7bb7c22241cc7d26094778996cc4d848d6cd4a4e526ada6    |
| nut_g_02.MOV        | 37,726,476   | 5958a5b58d6dab7ac764bfbb78708ba0a362eb12285d6f6006c07098ce561107    |
| nut_h_01.MOV        | 38,022,547   | a825128ae09b76102ee845a4c01ec4242f1e3466fe01603f683f8413b8bda117    |
| nut_h_02.MOV        | 37,532,070   | 14d8a5dc6d34c345187d74c965f5a12e1c4fc298fa771185f8d0cd0b00150b36    |
| nut_h_03.MOV        | 37,170,044   | b1121418c3baa93ce793e384b3d50ff4468f6fed8e0cd3efe5babf70f7401034    |
| screw_a_01.MOV      | 37,157,224   | fd2cc0fba2bdc693dfe0b3b7c10ded160b8c7c722e7c2d0e54282b69b6c0e36e    |
| screw_a_02.MOV      | 36,733,133   | 1b28db30978ea1822fa84afc6004053770d0bb33da78926b67e3480f7a8dd859    |
| screw_a_03.MOV      | 38,474,435   | 17b5824149c9a352bdd9c530fd4b4c5d05886affd5d08b8abcc26d6023a1a499    |
| screw_b_01.MOV      | 37,818,512   | 8d54b03b9b59a4ce3472fde99de2c81b93c00100a3b3da6893fc9dbcf1707d00    |
| screw_b_02.MOV      | 39,066,781   | ea1a49f93e5f678b47fdd0ca1d8605dc5ae5fdf09d5f06f718930163c7a10d1e    |
| screw_c_01.MOV      | 36,181,842   | 5c2a2760ddb667ef3f99f03340823fa267393e036456f872fd0c4e9ec3f4b1d6    |
| screw_c_02.MOV      | 37,579,066   | 7697db81c189583aeca3cddf66046d4efc5ac36e88821bced78752bd7ed81703    |
| screw_c_03.MOV      | 37,550,332   | dd3fad917ca4a776d1526f280d2c80759dd385fbcdd5d30579176dfa4f8b2f41    |
| screw_d_01.MOV      | 40,311,777   | f68705b6f0a4c59807cbd02e28b2397cffd95a31cab55353f0b3047cac8bc61f    |
| screw_d_02.MOV      | 36,427,892   | ea15c76d57ee4902300c0b0ce0e22878eb3782e5c9c368e305ef9dfcce9cf2de    |
| screw_e_01.MOV      | 37,102,616   | 40d857d1528238fe1aaee9a48a5900974a8a319eb654fe94a6d74e0a676d950f    |
| screw_e_02.MOV      | 38,033,136   | ff5e1e1a10f146903c2a4434d110af0e61cef979f9c5641ed90a5dd1f8699530    |
| screw_e_03.MOV      | 36,504,555   | 9480293a22f23bc90c50a4b89b1b155c5e92eaa83c0e9e9374d3fc8ff03c881b    |
| screw_f_01.MOV      | 38,075,453   | d170fa5ebfab6dcd5a8276402558a7556962584ee6964d8a0af2c41b41223552    |
| screw_f_02.MOV      | 37,986,579   | e9709f576dd7a10d3128fe2d99a29fb5c29ae830d0e07ecdf09fd5d6a334317e    |
| screw_f_03.MOV      | 36,994,444   | 8fbcc784170d79193882771917e95a96500e879832086f33d02707bf3b16db3b    |
| screw_g_01.MOV      | 37,601,219   | ae3249c7fd30f6652cad416ab00620bd3b22605e12896c276bdf89bb54b5b688    |
| screw_g_02.MOV      | 38,274,304   | e0ecf6f47b41e8e25d8c763f0f24628fd9abdf65ca5e747400f780564835877a    |
| screw_g_03.MOV      | 39,377,069   | ae25e731fd2c025574f38986929d2508169a4255b21b9249ea2361322c5fa317    |

## Components captured

- **screw**: variants `a, b, c, d, e, f, g` (7 variants x 2-3 takes each)
- **nut**:   variants `a, b, c, e, f, g, h` (7 variants x 2-3 takes each)
- **gear**:  variants `a, b` (2 variants x 2 takes each)

## How to obtain

Contact the repo owner -- these files live on the capture machine and are
shared via external storage (USB / NAS / cloud), not through git.
