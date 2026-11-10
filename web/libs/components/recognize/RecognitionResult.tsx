import { Alert, Chip, CircularProgress, Stack, Typography } from '@mui/material';
import { useTranslation } from 'next-i18next';
import Link from 'next/link';
import { useLandmark } from '../../hooks/useLandmark';
import { useLocalizedName } from '../../hooks/useLocalized';
import { Candidate, Recognition } from '../../types/landmark';
import LandmarkInfo from '../landmark/LandmarkInfo';

function CandidateChips({ candidates }: { candidates: Candidate[] }) {
	const localize = useLocalizedName();
	return (
		<Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
			{candidates.map((candidate) => (
				<Chip
					key={candidate.landmark_id}
					clickable
					component={Link}
					href={{ pathname: '/landmark/detail', query: { id: candidate.landmark_id } }}
					label={`${localize(candidate.name)} · ${candidate.score.toFixed(2)}`}
					variant="outlined"
				/>
			))}
		</Stack>
	);
}

export default function RecognitionResult({ result }: { result: Recognition }) {
	const { t } = useTranslation('common');
	const { data: landmark, error } = useLandmark(result.landmark_id);
	const [top, ...others] = result.candidates;
	const scoreLine = t('result.score', {
		score: top.score.toFixed(2),
		kind: result.score_kind,
		threshold: result.threshold.toFixed(2),
	});

	if (!result.recognized) {
		return (
			<section className="result result--unknown">
				<Typography variant="h3" component="h2">
					{t('result.unknown.title')}
				</Typography>
				<Typography color="text.secondary">{t('result.unknown.body')}</Typography>
				<Typography variant="subtitle2" className="result__label">
					{t('result.maybe')}
				</Typography>
				<CandidateChips candidates={result.candidates} />
				<Typography variant="caption" color="text.secondary" component="p">
					{scoreLine}
				</Typography>
			</section>
		);
	}

	return (
		<section className="result">
			<Typography variant="overline" color="secondary">
				{t('result.recognized')}
			</Typography>
			{error && <Alert severity="error">{t('error.load')}</Alert>}
			{!landmark && !error && <CircularProgress />}
			{landmark && <LandmarkInfo landmark={landmark} />}
			<Typography variant="caption" color="text.secondary" component="p">
				{scoreLine}
			</Typography>
			{others.length > 0 && (
				<>
					<Typography variant="subtitle2" className="result__label">
						{t('result.others')}
					</Typography>
					<CandidateChips candidates={others} />
				</>
			)}
		</section>
	);
}
