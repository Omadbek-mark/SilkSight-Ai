import { API_URL } from '../config';
import { LandmarkDetail, LandmarkSummary, ModelName, Recognition } from '../types/landmark';

export class ApiError extends Error {
	constructor(message: string, readonly status: number) {
		super(message);
	}
}

async function parse<T>(response: Response): Promise<T> {
	if (response.ok) return response.json() as Promise<T>;
	let detail = response.statusText;
	try {
		const body = await response.json();
		if (typeof body.detail === 'string') detail = body.detail;
	} catch {
		// Non-JSON error body: keep the status text.
	}
	throw new ApiError(detail, response.status);
}

export const fetchLandmarks = async (): Promise<LandmarkSummary[]> => parse(await fetch(`${API_URL}/api/landmarks`));

export const fetchLandmark = async (id: string): Promise<LandmarkDetail> =>
	parse(await fetch(`${API_URL}/api/landmarks/${encodeURIComponent(id)}`));

export async function recognize(file: File, model: ModelName): Promise<Recognition> {
	const body = new FormData();
	body.append('file', file);
	return parse(await fetch(`${API_URL}/api/recognize?model=${model}`, { method: 'POST', body }));
}
