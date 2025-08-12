import numpy as np
import matplotlib.pyplot as plt

# data = np.load('./double_ramp_configuration/double_ramp_npz_files_clamped/double_ramp_0.049_0.0636_ma_2.511_pres_193365_interpolated_arrays.npz')
# # print(data.files)

# for key in data.files:
#     array = data[key]
#     print(f"\nKey: {key}")
#     print(f"Shape: {array.shape}")
#     print(f"Data type: {array.dtype}")
#     print(f"First few elements:\n{array[:5]}")

# density = data['density']

# plt.imshow(density, cmap='viridis')  # or 'gray' for traditional grayscale - origin='lower' -
# # plt.colorbar(label='Density')
# # plt.title('Density Map')
# # plt.xlabel('X')
# # plt.ylabel('Y')
# plt.axis('off')  # Hide axes
# plt.savefig('./double_ramp_configuration/cv_playground/density_plot.png', dpi=100, bbox_inches='tight', pad_inches=0)  # Save before showing 
# plt.show()
data_n = 'double_ramp_0.0105_0.0232_ma_7.125_pres_91820_interpolated_arrays'
data = np.load('./double_ramp_configuration/inputs/double_ramp_npz_files_clamped/'+data_n+'.npz')

density = data['density']
height, width = density.shape  # should be 256x256

# Set figure size to match the desired pixel size (in inches = pixels / DPI)
dpi = 100
figsize = (width / dpi, height / dpi)

# Create figure with specific size
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])  # full-figure axes
ax.set_axis_off()
fig.add_axes(ax)

# Show image
ax.imshow(density, cmap='viridis', aspect='auto')
fig.savefig(
    './double_ramp_configuration/X_dummy_code_playground/png_input/density_plot.png', 
    dpi=dpi,
    bbox_inches='tight',
    pad_inches=0
)
plt.close(fig)

