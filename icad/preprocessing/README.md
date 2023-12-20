# Image preprocessing

The image preprocessing pipeline consists of the following steps:
* Atlas resampling to median spacing of the dataset.
* Affine registration of the CTA to the resampled atlas.
* Affine registration of the NCCT to the registered CTA.
* Registration of the vessel and thrombus masks, obtained by the [Arterial](https://www.sciencedirect.com/science/article/pii/S0895611122001409) vascular segmentation and manual segmentation, respectively, using the affine transform from the CTA registration.

## Resampled atlas to median spacing

For this task, we forsee that comparing radiological information from the contralateral side to the occlusion. Thus, it can be interesting to register all images to a common space. We chose the [152 MNI brain](https://nist.mni.mcgill.ca/atlases/) atlas for reference, of `1 x 1 x 1` mm^3. We computed the median spacing of the dataset and resampled the original MNI atlas to the median spacing of the dataset (roughly `[0.43, 0.43, 0.4]` mm^3).

## NCCT and CTA registration
