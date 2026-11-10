import { useTranslation } from 'next-i18next';
import { assetUrl } from '../../config';
import { Photo } from '../../types/landmark';

export default function LandmarkPhoto({ photo, alt }: { photo: Photo | null; alt: string }) {
	const { t } = useTranslation('common');
	if (!photo) return <div className="landmark-photo landmark-photo--empty" />;
	return (
		<figure className="landmark-photo">
			{/* eslint-disable-next-line @next/next/no-img-element -- served by the local API, not next/image */}
			<img src={assetUrl(photo.url)} alt={alt} loading="lazy" />
			<figcaption>
				<a href={photo.source_url} target="_blank" rel="noopener noreferrer">
					{t('landmark.photoBy', { author: photo.author, license: photo.license })}
				</a>
			</figcaption>
		</figure>
	);
}
