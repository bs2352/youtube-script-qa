export type SummaryRequestBody = {
    vid: string;
}

export type SummaryResponseBody = {
    title: string;
    author: string;
    lengthSeconds: number;
    url: string;
    concise: string;
    detail: string[];
    topic: TopicType[];
    keyword: string[];
}

export type TopicType = {
    title: string;
    abstract: string[];
}
