import HotelIcon from '@mui/icons-material/HotelOutlined';
import PlaceIcon from '@mui/icons-material/PlaceOutlined';
import { Button, Chip, Stack, Typography } from '@mui/material';
import { useTranslation } from 'next-i18next';
import dynamic from 'next/dynamic';
import { useLocalizedName } from '../../hooks/useLocalized';
import { LandmarkDetail } from '../../types/landmark';
import LandmarkPhoto from './LandmarkPhoto';
import WikipediaText from './WikipediaText';

const LandmarkMap = dynamic(() => import('./LandmarkMap'), { ssr: false });

// Buildings need street level; lakes and canyons need their surroundings.
const MAP_ZOOM = { architecture: 16, monument: 16, nature: 11 };

export default function LandmarkInfo({ landmark }: { landmark: LandmarkDetail }) {
	const { t } = useTranslation('common');
	const localize = useLocalizedName();
	const name = localize(landmark.name);
	const place = [landmark.city, t(`country.${landmark.country}`)].filter(Boolean).join(', ');

	return (
		<article className="landmark-info">
			<div className="landmark-info__media">
				<LandmarkPhoto photo={landmark.photo} alt={name} />
			</div>
			<div className="landmark-info__body">
				<Typography variant="h3" component="h2">
					{name}
				</Typography>
				<Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" className="landmark-info__meta">
					<Chip icon={<PlaceIcon />} label={place} variant="outlined" />
					<Chip label={t(`category.${landmark.category}`)} variant="outlined" />
					{landmark.unesco_site_qid && <Chip label={t('landmark.unesco')} color="primary" variant="outlined" />}
				</Stack>
				<WikipediaText landmark={landmark} />
			</div>
			<section className="landmark-info__location">
				<Typography variant="h4" component="h3">
					{t('landmark.location')}
				</Typography>
				<LandmarkMap
					zoom={MAP_ZOOM[landmark.category]}
					latitude={landmark.latitude}
					longitude={landmark.longitude}
					label={name}
				/>
				{landmark.hotelia_url ? (
					<Button
						variant="contained"
						color="secondary"
						startIcon={<HotelIcon />}
						href={landmark.hotelia_url}
						target="_blank"
						rel="noopener noreferrer"
					>
						{t('landmark.hotels', { city: landmark.city })}
					</Button>
				) : (
					<Typography variant="body2" color="text.secondary">
						{t('landmark.noHotels')}
					</Typography>
				)}
			</section>
		</article>
	);
}
