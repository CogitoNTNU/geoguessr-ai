import React, { useMemo, useState } from "react";
import "./MainPage.scss";

type Coordinates = { lat: number; lng: number } | null;

const MainPage: React.FC = () => {
  const [image, setImage] = useState<File | null>(null);
  const [coords, setCoords] = useState<Coordinates>(null);

  const imageUrl = useMemo(
    () => (image ? URL.createObjectURL(image) : ""),
    [image]
  );

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setImage(file);
  };

  const mockPredict = async () => {
    setCoords(null);
    // Mock API: pretend to send image, return random lat/lng
    await new Promise((r) => setTimeout(r, 600));
    const lat = +(Math.random() * 180 - 90).toFixed(4);
    const lng = +(Math.random() * 360 - 180).toFixed(4);
    setCoords({ lat, lng });
  };

  const mapSrc = coords
    ? `https://www.google.com/maps?q=${coords.lat},${coords.lng}&z=4&output=embed`
    : "";

  return (
    <div className="mainPage">
      <div className="card">
        <h1>Geoguessr AI</h1>
        <p>Upload a photo, then get a guessed location.</p>
        <label className="upload">
          <input type="file" accept="image/*" onChange={handleUpload} />
          <span>{image ? image.name : "Choose an image"}</span>
        </label>
        {image && (
          <div className="preview">
            <img src={imageUrl} alt="preview" />
          </div>
        )}
        <button className="action" onClick={mockPredict} disabled={!image}>
          Guess Location
        </button>
      </div>

      {coords && (
        <div className="map">
          <iframe
            title="map"
            src={mapSrc}
            loading="lazy"
            referrerPolicy="no-referrer-when-downgrade"
          />
          <div className="coords">
            lat {coords.lat}, lng {coords.lng}
          </div>
        </div>
      )}
    </div>
  );
};

export default MainPage;
