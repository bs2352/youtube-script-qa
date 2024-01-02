export type SummaryRequestBody = {
    vid: string;
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
    topic: TopicType[];
    keyword: string[];
}

export type SummaryResponseBody = {
    vid: string;
    summary: SummaryType;
}

export type TopicType = {
    title: string;
    abstract: string[];
}

export type SampleVideoInfo = {
    vid: string;
    title: string;
    author: string;
    lengthSeconds: number;
}

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