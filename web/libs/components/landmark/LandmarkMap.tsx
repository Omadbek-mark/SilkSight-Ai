import { CircleMarker, MapContainer, TileLayer, Tooltip } from 'react-leaflet';

interface Props {
	zoom: number;
	latitude: number;
	longitude: number;
	label: string;
}

// Loaded only in the browser (Leaflet needs `window`): import with next/dynamic, ssr: false.
export default function LandmarkMap({ zoom, latitude, longitude, label }: Props) {
	return (
		<MapContainer center={[latitude, longitude]} zoom={zoom} scrollWheelZoom={false} className="landmark-map">
			<TileLayer
				attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
				url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
			/>
			<CircleMarker center={[latitude, longitude]} radius={10} pathOptions={{ color: '#b8622a', fillOpacity: 0.8 }}>
				<Tooltip direction="top" offset={[0, -8]} permanent>
					{label}
				</Tooltip>
			</CircleMarker>
		</MapContainer>
	);
}
