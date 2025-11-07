from torch.utils.data import DataLoader
from data import ImageDataset


def train_model(data, labels, num_epochs: int):
    """
    Train the model
    Data = bilder
    Labels = tilhørende metadata med følgende felter:
    lat: float
    lon: float
    heading: int
    location_id: str
    pano_id: str
    capture_date: str
    batch_date: str
    """
    dataset = ImageDataset(data, labels)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    """
    Hva som må gjøres (roughly)
    * Sette scriptet til å kjøre på GPU: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    * Initialisere SuperGuessr-modellen
    * Sette modellen i treningsmodus ved å kalle model.train()
    * Initialisere en optimizer 
    """

    for epoch in range(num_epochs):
        for batch_images, batch_metadata in train_dataloader: # Load data in batches
            pass
        """
        * Hente 4 bilder av gangen
        * Lage variabel for lat, long hentet fra batch_metadata
        * Kalle på modellen og mate de 4 bildene inn her, få ut loss og topk geocells (geocell_topk variabel)
        * batch_images = batch_images.to(device)  # [B, 3, H, W]
        * For hver batch lager vi en liste med alle (lat, long)-par, altså fasit-koordinatene.
        * Vi lager så en liste med geocell_id-ene (eller noe annet, f.eks. centroiden) til geocellene disse koordinatene faller inni, kall den labels_ground_truth
        * Vi må kjøre optimizer.zero_grad(set_to_none=True) for å fjerne tidligere gradient-buffers
        * Vi kjører forward-pass gjennom modellen og setter labels_clf=labels_ground_truth, og får ut en loss.
        * Vi kjører loss.backward() for å oppdatere vektene til modellen
        * Optimizer.step()
        * Repeat
        """
