# notes to refiner:

Task:
Create the proto data

init:

protos:
index: geocell_id = ["embedding":tensor]

proto_df: f.eks {
"indices": [12, 34, 56], # dataset row indices in self.dataset["train"]
"count": 3, # number of points in the cluster
"lng": \<mean_lng>, # cluster mean or representative longitude
"lat": \<mean_lat>, # cluster mean or representative latitude
}

index_json =
{
"indices": ["1", "2",...]
}

dataset: Dataset from S3 bucket?
{
lat:
lon:
image:
pano_id:
location_id:
capture_date:
}

proto_df: hvordan skal den se ut???
den får mere data om hvert punkt

## init end.

from preprocessing import haversine: Func in geo_utils for haversine distance (tensor, tensor) -> tensor (km)

Helper functions:

cosine_sim: (self,matrix tensor, vector tensor) -> Tensor (cosine sim)

euclidian_dis: (self,matrix tensor, vector tensor) -> Tensor (distance)

temperature_softmax: (self, tensor) -> tensor (softmax with learnable parapeters)

compute_protos_for_cell: Computes embedding and geo prototypes

se mer på:
self.proto_df

\_get_prototypes: gets the embedings for data related to a cell

forward:
embedding: embeddings of image (output from CLIP or TinyViT)
pred_LLH: Liste med lat lon punkter (høyest oppe er mest sannsynlig)
candidate_cells: index of the top five guesses
candidate_probs: values to the top five guesses

trenger csv fil med alle centroids til geocells. (lat,lon)
trenger haversine func

prototypes: points to a cluster

proto_path
dataset_path

index_json

```
# print(len(mang.geocells))
# print(len(mang.geocells["Japan"]))
# print(len(mang.geocells["Japan"]["Tokyo"]))
# print(len(mang.geocells["Japan"]["Tokyo"][0]))
# print(mang.geocells["Japan"]["Tokyo"][0].points[0].name) #["latitude": 35.6895, "longitude": 139.6917, "Name: idx 0, dtype: float64"] - .name get the index of the point (pandas Series)
# print(mang.geocells["Japan"]["Tokyo"][0].points[1])
# print(mang.geocells["Japan"]["Tokyo"][0].points[2])

mang.geocells["Japan"]["Tokyo"][0] = geocell
```

702: -42.344647, -63.957823 (Argentina)

for cluster_id, cluster_data in cell.clusters.items():
print(f"Cluster ID: {cluster_id}")
print(f"Number of points in cluster: {len(cluster_data['points'])}")
print(f"Cluster centroid: {cluster_data['centroid']}")

```
# Access individual points in the cluster
for point in cluster_data['points']:
    print(f"Point name: {point.name}")
    print(f"Point latitude: {point['latitude']}, longitude: {point['longitude']}")
```

Jens mekker embeddings:
"Locationid:str, heading:int, photo:Path?, emb_clip:Tensor, emb_tiny_vit:Tensor"
