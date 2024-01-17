export type SummaryRequestBody = {
    vid: string;
    refresh?: boolean;
}

export type DetailSummaryType = {
    start: number;
    text: string;
}

export type SummaryType = {
    title: string;
    author: string;
    lengthSeconds: number;
    url: string;
    concise: string;
    detail: DetailSummaryType[];
    agenda: AgendaType[];
    keyword: string[];
    topic: TopicType[];
}

export type SummaryResponseBody = {
    vid: string;
    summary: SummaryType;
}

export type AgendaType = {
    title: string;
    subtitle: string[];
    time: (string[])[];
}

export type TopicType = {
    topic: string;
    time: string[];
}

export type SampleVideoInfo = {
    vid: string;
    title: string;
    author: string;
    lengthSeconds: number;
    url: string;
}

export type VideoInfoType = SampleVideoInfo;

export type QaRequestBody = {
    vid: string;
    question?: string;
    query?: string;
    ref_sources: number;
}

export type QaAnswerSource = {
    id: string;
    score: number;
    time: string;
    source: string;
}

export type QaResponseBody = {
    vis: string;
    question?: string;
    query?: string;
    answer?: string;
    sources: QaAnswerSource[]
}

export type TranscriptRequestBody = {
    vid: string;
}

export type TranscriptType = {
    id: string;
    text: string;
    start: number;
    duration: number;
    overlap: number;
}

export type TranscriptResponseBody = {
    vid: string;
    transcripts: TranscriptType[];
}