import { Alert, Button, CircularProgress, Container } from '@mui/material';
import { GetStaticProps } from 'next';
import { useTranslation } from 'next-i18next';
import { serverSideTranslations } from 'next-i18next/serverSideTranslations';
import Link from 'next/link';
import { useRouter } from 'next/router';
import LandmarkInfo from '../../libs/components/landmark/LandmarkInfo';
import withLayout from '../../libs/components/layout/withLayout';
import { useLandmark } from '../../libs/hooks/useLandmark';

function LandmarkDetailPage() {
	const { t } = useTranslation('common');
	const router = useRouter();
	const id = router.isReady && typeof router.query.id === 'string' ? router.query.id : null;
	const { data, error } = useLandmark(id);
	return (
		<Container maxWidth="lg" className="landmark-detail-page">
			<Button component={Link} href="/landmark">
				← {t('landmark.back')}
			</Button>
			{error && <Alert severity="warning">{t('landmark.notFound')}</Alert>}
			{!data && !error && <CircularProgress />}
			{data && <LandmarkInfo landmark={data} />}
		</Container>
	);
}

export const getStaticProps: GetStaticProps = async ({ locale }) => ({
	props: { ...(await serverSideTranslations(locale ?? 'en', ['common'])) },
});

export default withLayout(LandmarkDetailPage);
