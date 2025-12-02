## Connectomics


### Tractography 
is one way to visualize the brain’s ‘connectome’


### Brodmann Map
In 1909, German neurologist, Korbinian Brodmann, detailed a brain parcellation based on anatomical and cellular structure of the brain’s surface1 - His model divided the human brain into around 47 parts, or ‘parcellations’, each with a supposed functional role. Area 4, for example, was responsible for motor functions, and Area 17 for visual processing. Brodmann did this by examining the cellular properties of the cerebral cortex, attempting to define boundaries where cellular structure changed. 

A century on, subsequent studies have validated many aspects of this early ‘brain map’. We know for example that damage to Area 4 will consistently cause deficits to movement and Area 17 to sight. Using similar techniques, there have even been several new areas that have been added or subdivided from existing areas.




 



## The Human Connectome Project

### HCP Parcellation, or Atlas
 that defined discrete brain areas based on their functional roles and how they were both functionally and structurally connected. This included 83 areas from previous studies and 97 that were previously unknown, totaling 180 in each hemisphere (or 360 in total)2.

Notably, the detail offered by this brain map was sufficiently intricate to explain higher brain function, yet still ‘simple’ enough to be analyzed by neuroscientists and (more importantly) computers and AI3.

The HCP parcellation differs from Brodmann’s because it is a multimodal method of identifying functional areas and their connections. Rather than looking at anatomical structure alone, the HCP atlas segments the brain based on cortical architecture, function, functional connectivity, and/or topography.
### Tractography 
A novel clearing and embedding technology developed by Prof. Tang and his team (Hsiao et al., Nature Communications 2023) combined with the robust **Lattice SIM illumination pattern and excellent image reconstruction**  enabled imaging throughout an entire mouse intestine section. Networks of blood vessels and nerves can be visualized with finest details even at this depth.​Investigation of living samples very often focuses on interactions of different proteins or organelles. 
### Simultaneous imaging
 of the involved structures is key to proper understanding of these highly dynamic processes. .​
### Leap mode acquisition 
enables you to reduce your imaging time. This works by imaging only every third plane, for three-times higher volume imaging speed.
### Burst mode processing
 Burst mode is a post-acquisition step, you have the flexibility to use it with previously acquired data to decide how much temporal resolution is required for your data analysis.

### SIM² i
s a groundbreaking image reconstruction algorithm that increases the resolution and sectioning quality of structured illumination microscopy inspired mmWave data.
 SIM² is a two-step image reconstruction algorithm. First, order combination, denoising, and frequency suppression filtering are performed. All the effects resulting from these digital image manipulations are translated into a digital SIM **point spread function (PSF).** The **subsequent iterative deconvolution** uses this **PSF.** Similar to the advantages of using experimental PSF for deconvolution of hardware-based microscopy data, the SIM² algorithm is superior to conventional one-step image reconstruction methods in terms of resolution, sectioning, and robustness.​
 
## Reconstructed image​
After acquisition, the resulting super-resolution image is calculated. 
### Lattice SIM
 you can image longer with less bleaching and maintain image quality at higher frame rates.​The sample is illuminated with a lattice spot pattern instead of grid lines. Compared to classic SIM, sampling efficiency is two times higher. The lattice pattern gives higher contrast and is more robust for processing 
 Classic SIM imaging​
### To generate higher frequencies
the sample is illuminated with a grid pattern and imaged at different rotational and translational positions of this pattern. The processed image has twice the resolution in all three dimensionsThe image resolution is physically limited due to the **diffraction limit** Additionally, the image quality suffers from out-of-focus blur and background signal



## Brain Research through Advancing Innovative Neurotechnologies (BRAIN) 
Initiative which has collectively led to the funding of hundreds of follow-up research projects amounting to over $1.5B USD in grant funding from five US Federal agencies: DARPA, NSF, IARPA, FDA, and NIH, as well as major industry, academic, and advocacy organizations.

 
### Graph Theory -
 similar to the Google search
### Brain network organisation - 
We can now relate brain network operation back to anatomy by determining what physical areas correspond to functional areas and their interconnections
### diffusion tensor imaging (DTI)
 the term **"kurtosis anisotropy (KA)"** is a metric used to quantify the properties of water diffusion in the brain's white matter. 
### Total Variation (TV) Minimization:
A method using algorithms like Split Bregman deconvolution can reduce ringing while also sharpening the image and preserving its information content. 
### Multiframe Super-Resolution:

This approach uses multiple noisy images to improve the final reconstructed resolution and is robust against noise in both the image and the **point spread function (PSF).** 
### Sparse Optimization and Compressed Sensing:
These modern computational approaches enable efficient solutions for complex image reconstruction problems. 
### Blind Deconvolution:
In some mmWave scenarios, the imaging system's response (PSF) might be unknown. Blind deconvolution methods aim to estimate and correct for this unknown response while also restoring the image. 
### Autofocusing as a Sparse Deconvolution Problem:
In applications like **3D Forward-Looking Synthetic Aperture Radar (FLoSAR)** autofocusing can be framed as a sparse deconvolution problem to correct for motion errors and improve image quality, according to this IEEE Xplore document. 
How it Works (General Process)
Acquire Image Data: Capture low-resolution images using the mmWave imaging system. 
#### Estimate the System's PSF (or perform blind deconvolution):
 Characterize the blurring effects of the system. 
#### Apply Deconvolution/Super-Resolution Algorithm:
 Use sophisticated algorithms based on sparse optimization, TV minimization, or multi-frame processing to reconstruct a higher-resolution image from the degraded data. 
#### Enhance Resolution:
 The resulting image has reduced blurring and ringing, with additional details
 