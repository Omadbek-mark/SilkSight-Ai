import { Typography } from '@mui/material';
import { useTranslation } from 'next-i18next';
import Link from 'next/link';
import { assetUrl } from '../../config';
import { useLocalizedName } from '../../hooks/useLocalized';
import { LandmarkSummary } from '../../types/landmark';

export default function LandmarkCard({ landmark }: { landmark: LandmarkSummary }) {
	const { t } = useTranslation('common');
	const name = useLocalizedName()(landmark.name);
	return (
		<Link href={{ pathname: '/landmark/detail', query: { id: landmark.id } }} className="landmark-card">
			<div className="landmark-card__image">
				{landmark.photo && (
					// eslint-disable-next-line @next/next/no-img-element -- served by the local API
					<img src={assetUrl(landmark.photo.url)} alt={name} loading="lazy" />
				)}
			</div>
			<div className="landmark-card__text">
				<Typography fontWeight={600}>{name}</Typography>
				<Typography variant="body2" color="text.secondary">
					{[landmark.city, t(`country.${landmark.country}`)].filter(Boolean).join(', ')}
				</Typography>
			</div>
		</Link>
	);
}
