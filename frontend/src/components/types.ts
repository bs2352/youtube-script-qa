export type SummaryRequestBody = {
    vid: string;
}

export type SummaryType = {
    title: string;
    author: string;
    lengthSeconds: number;
    url: string;
    concise: string;
    detail: string[];
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
    question: string;
    ref_source: number;
}

export type QaAnswerSource = {
    score: number;
    time: string;
    source: string;
}

export type QaResponseBody = {
    vis: string;
    question: string;
    answer: string;
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