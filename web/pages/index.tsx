import { Container, Typography } from '@mui/material';
import { GetStaticProps } from 'next';
import { useTranslation } from 'next-i18next';
import { serverSideTranslations } from 'next-i18next/serverSideTranslations';
import { useEffect, useState } from 'react';
import { ApiError, recognize } from '../libs/api/silksight';
import withLayout from '../libs/components/layout/withLayout';
import RecognitionResult from '../libs/components/recognize/RecognitionResult';
import UploadPanel from '../libs/components/recognize/UploadPanel';
import { API_URL } from '../libs/config';
import { useLandmarks } from '../libs/hooks/useLandmark';
import { ModelName, Recognition } from '../libs/types/landmark';

function RecognizePage() {
	const { t } = useTranslation('common');
	const { data: landmarks } = useLandmarks();
	const [file, setFile] = useState<File | null>(null);
	const [previewUrl, setPreviewUrl] = useState<string | null>(null);
	const [model, setModel] = useState<ModelName>('knn');
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [result, setResult] = useState<Recognition | null>(null);

	useEffect(() => {
		if (!file) return setPreviewUrl(null);
		const url = URL.createObjectURL(file);
		setPreviewUrl(url);
		return () => URL.revokeObjectURL(url);
	}, [file]);

	// A new photo or model makes the previous answer stale.
	const choose = (next: File | null) => {
		setFile(next);
		setResult(null);
		setError(null);
	};
	const changeModel = (next: ModelName) => {
		setModel(next);
		setResult(null);
	};

	const submit = async () => {
		if (!file || busy) return;
		setBusy(true);
		setError(null);
		try {
			setResult(await recognize(file, model));
		} catch (caught) {
			setResult(null);
			setError(
				caught instanceof ApiError
					? t('error.server', { detail: caught.message })
					: t('error.network', { url: API_URL }),
			);
		} finally {
			setBusy(false);
		}
	};

	return (
		<Container maxWidth="lg" className="recognize-page">
			<div className="recognize-page__hero">
				<Typography variant="h1">{t('hero.title')}</Typography>
				<Typography color="text.secondary" className="recognize-page__subtitle">
					{t('hero.subtitle', { count: landmarks?.length ?? 23 })}
				</Typography>
			</div>
			<UploadPanel
				previewUrl={previewUrl}
				busy={busy}
				error={error}
				model={model}
				onModelChange={changeModel}
				onFile={choose}
				onSubmit={submit}
				onReset={() => choose(null)}
			/>
			{result && <RecognitionResult result={result} />}
		</Container>
	);
}

export const getStaticProps: GetStaticProps = async ({ locale }) => ({
	props: { ...(await serverSideTranslations(locale ?? 'en', ['common'])) },
});

export default withLayout(RecognizePage);
