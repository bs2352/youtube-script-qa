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

