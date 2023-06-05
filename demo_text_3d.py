from imagebind.imagebind_model import ModalityType
from utils.data_transform import load_and_transform_text
import models.PointBind_models as models
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_pc_features(model, pc):     # For 3D feature encoding
    pc_features = model.encode_pc(pc)
    pc_features = model.bind.modality_head_point(pc_features)
    pc_features = model.bind.modality_postprocessor_point(pc_features)
    return pc_features

# Instantiate model.
model = models.PointBind_I2PMAE()
state_dict = torch.load("./ckpts/pointbind_i2pmae.pt", map_location='cpu')
model.load_state_dict(state_dict, strict=True)
model.eval().to(device)

# Encode text features.
text_list = ['An airplane', 'A car', 'A toilet']
inputs = {
    ModalityType.TEXT: load_and_transform_text(text_list, device),
}
with torch.no_grad():
    text_features = model.bind(inputs)[ModalityType.TEXT]

# Encode 3D point cloud features.
point_paths = ["examples/airplane.pt", "examples/car.pt", "examples/toilet.pt"]
points = []
for point_path in point_paths:
    point = torch.load(point_path)
    points.append(point)
points = torch.stack(points, dim=0).to(device)
pc_features = get_pc_features(model, points)

# Calculate similarity matrix.
logits = torch.softmax(pc_features @ text_features.t(), dim=-1)
print(
    "Text x Point Cloud: \n",
    logits
)
