import { useEffect, useState } from 'react';
import { fetchLandmark, fetchLandmarks } from '../api/silksight';
import { LandmarkDetail, LandmarkSummary } from '../types/landmark';

interface Loadable<T> {
	data: T | null;
	error: boolean;
}

function useLoad<T>(load: (() => Promise<T>) | null, key: string): Loadable<T> {
	const [state, setState] = useState<Loadable<T>>({ data: null, error: false });
	useEffect(() => {
		if (!load) return;
		let active = true;
		setState({ data: null, error: false });
		load()
			.then((data) => active && setState({ data, error: false }))
			.catch(() => active && setState({ data: null, error: true }));
		return () => {
			active = false;
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps -- reload only when the key changes
	}, [key]);
	return state;
}

export const useLandmark = (id: string | null) =>
	useLoad<LandmarkDetail>(id ? () => fetchLandmark(id) : null, id ?? '');

export const useLandmarks = () => useLoad<LandmarkSummary[]>(fetchLandmarks, 'all');
