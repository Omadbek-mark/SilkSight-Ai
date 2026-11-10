import { Alert, CircularProgress, Container, Typography } from '@mui/material';
import { GetStaticProps } from 'next';
import { useTranslation } from 'next-i18next';
import { serverSideTranslations } from 'next-i18next/serverSideTranslations';
import LandmarkCard from '../../libs/components/landmark/LandmarkCard';
import withLayout from '../../libs/components/layout/withLayout';
import { useLandmarks } from '../../libs/hooks/useLandmark';

function LandmarksPage() {
	const { t } = useTranslation('common');
	const { data, error } = useLandmarks();
	return (
		<Container maxWidth="lg" className="landmarks-page">
			<Typography variant="h2" component="h1">
				{t('landmarks.title')}
			</Typography>
			{data && <Typography color="text.secondary">{t('landmarks.subtitle', { count: data.length })}</Typography>}
			{error && <Alert severity="error">{t('error.load')}</Alert>}
			{!data && !error && <CircularProgress />}
			<div className="landmarks-page__grid">
				{data?.map((landmark) => (
					<LandmarkCard key={landmark.id} landmark={landmark} />
				))}
			</div>
		</Container>
	);
}

export const getStaticProps: GetStaticProps = async ({ locale }) => ({
	props: { ...(await serverSideTranslations(locale ?? 'en', ['common'])) },
});

export default withLayout(LandmarksPage);
