export type Locale = 'en' | 'uz';

export interface LocalizedName {
	en: string;
	uz?: string;
}

export interface Photo {
	url: string;
	title: string;
	author: string;
	license: string;
	license_url: string;
	source_url: string;
}

export interface LandmarkSummary {
	id: string;
	name: LocalizedName;
	country: string;
	city: string | null;
	category: 'architecture' | 'monument' | 'nature';
	latitude: number;
	longitude: number;
	photo: Photo | null;
}

export interface WikipediaText {
	language: string;
	title: string;
	url: string;
	revision_id: number;
	text: string;
	license: string;
	license_url: string;
}

export interface LandmarkDetail extends LandmarkSummary {
	summary: Partial<Record<Locale, WikipediaText>>;
	unesco_site_qid: string | null;
	wikipedia: Partial<Record<string, string>>;
	wikidata_url: string;
	hotelia_url: string | null;
}

export type ModelName = 'knn' | 'classifier';

export interface Candidate {
	landmark_id: string;
	score: number;
	name: LocalizedName;
}

export interface Recognition {
	model: ModelName;
	score_kind: string;
	threshold: number;
	recognized: boolean;
	landmark_id: string | null;
	landmark: LandmarkSummary | null;
	candidates: Candidate[];
}
