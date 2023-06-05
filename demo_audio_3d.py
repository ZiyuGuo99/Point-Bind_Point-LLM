from imagebind.imagebind_model import ModalityType
from utils.data_transform import load_and_transform_audio_data
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

# Load and transform data.
audio_paths = ["examples/airplane_audio.wav", "examples/car_audio.wav", "examples/toilet_audio.wav"]
inputs = {
    ModalityType.AUDIO: load_and_transform_audio_data(audio_paths, device),
}
point_paths = ["examples/airplane.pt", "examples/car.pt", "examples/toilet.pt"]
points = []
for point_path in point_paths:
    point = torch.load(point_path)
    points.append(point)
points = torch.stack(points, dim=0).to(device)

# Encode audio features and 3D point cloud features.
with torch.no_grad():
    embeddings = model.bind(inputs)
    audio_features = embeddings[ModalityType.AUDIO]
    pc_features = get_pc_features(model, points)

# Calculate similarity matrix.
logits = torch.softmax(pc_features @ audio_features.t(), dim=-1)
print(
    "Audio x Point Cloud: \n",
    logits
)
