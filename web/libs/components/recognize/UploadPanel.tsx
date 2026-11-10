import CloudUploadIcon from '@mui/icons-material/CloudUploadOutlined';
import { Alert, Button, CircularProgress, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import { useTranslation } from 'next-i18next';
import { DragEvent, useRef, useState } from 'react';
import { ACCEPTED_TYPES, MAX_UPLOAD_BYTES } from '../../config';
import { ModelName } from '../../types/landmark';

interface Props {
	previewUrl: string | null;
	busy: boolean;
	error: string | null;
	model: ModelName;
	onModelChange: (model: ModelName) => void;
	onFile: (file: File) => void;
	onSubmit: () => void;
	onReset: () => void;
}

export default function UploadPanel(props: Props) {
	const { t } = useTranslation('common');
	const input = useRef<HTMLInputElement>(null);
	const [dragging, setDragging] = useState(false);
	const [localError, setLocalError] = useState<string | null>(null);

	const accept = (file: File | undefined) => {
		if (!file) return;
		if (!ACCEPTED_TYPES.includes(file.type)) return setLocalError(t('error.type'));
		if (file.size > MAX_UPLOAD_BYTES) return setLocalError(t('error.size'));
		setLocalError(null);
		props.onFile(file);
	};
	const onDrop = (event: DragEvent) => {
		event.preventDefault();
		setDragging(false);
		accept(event.dataTransfer.files[0]);
	};
	const error = localError ?? props.error;

	return (
		<section className="upload">
			{props.previewUrl ? (
				<div className="upload__preview">
					{/* eslint-disable-next-line @next/next/no-img-element -- local object URL */}
					<img src={props.previewUrl} alt={t('upload.preview')} />
				</div>
			) : (
				<div
					className={`upload__drop${dragging ? ' is-dragging' : ''}`}
					onDragOver={(event) => {
						event.preventDefault();
						setDragging(true);
					}}
					onDragLeave={() => setDragging(false)}
					onDrop={onDrop}
				>
					<CloudUploadIcon className="upload__icon" />
					<Typography fontWeight={600}>{t('upload.drop')}</Typography>
					<Typography variant="body2" color="text.secondary">
						{t('upload.or')}
					</Typography>
					<Button variant="outlined" onClick={() => input.current?.click()}>
						{t('upload.choose')}
					</Button>
					<Typography variant="caption" color="text.secondary">
						{t('upload.hint')}
					</Typography>
				</div>
			)}
			<input
				ref={input}
				type="file"
				accept={ACCEPTED_TYPES.join(',')}
				hidden
				onChange={(event) => {
					accept(event.target.files?.[0]);
					event.target.value = '';
				}}
			/>

			<div className="upload__controls">
				<div>
					<Typography variant="caption" color="text.secondary" component="p">
						{t('model.label')}
					</Typography>
					<ToggleButtonGroup
						size="small"
						exclusive
						value={props.model}
						disabled={props.busy}
						onChange={(_, value: ModelName | null) => value && props.onModelChange(value)}
					>
						<ToggleButton value="knn">{t('model.knn')}</ToggleButton>
						<ToggleButton value="classifier">{t('model.classifier')}</ToggleButton>
					</ToggleButtonGroup>
				</div>
				<div className="upload__actions">
					{props.previewUrl && (
						<Button onClick={props.onReset} disabled={props.busy}>
							{t('upload.another')}
						</Button>
					)}
					<Button
						variant="contained"
						disabled={!props.previewUrl || props.busy}
						onClick={props.onSubmit}
						startIcon={props.busy ? <CircularProgress size={18} color="inherit" /> : undefined}
					>
						{props.busy ? t('upload.working') : t('upload.recognize')}
					</Button>
				</div>
			</div>
			{error && <Alert severity="error">{error}</Alert>}
			<Typography variant="caption" color="text.secondary" component="p" className="upload__privacy">
				{t('upload.privacy')}
			</Typography>
		</section>
	);
}
