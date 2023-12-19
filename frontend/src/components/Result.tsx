import { SummaryResponseBody, SummaryType } from "./types"
import './Result.css'

function s2hms (seconds: number) {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0')
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0')
    const s = Math.floor(seconds % 60).toString().padStart(2, '0')
    return `${h}:${m}:${s}`
}


interface ResultProps {
    summary: SummaryResponseBody
}

interface ResultTableProps {
    summary: SummaryType
}

export function Result ({
    summary,
}: ResultProps) {

    const ResultTable = ({
        summary
    } : ResultTableProps) : JSX.Element => {
        return (
            <table className="div-table-resulttable">
                <tbody>
                    <tr>
                        <td className="div-table-td-title-resulttable">タイトル</td>
                        <td>{summary.title}</td>
                    </tr>
                    <tr>
                        <td className="div-table-td-title-resulttable">チャンネル名</td>
                        <td>{summary.author}</td>
                    </tr>
                    <tr>
                        <td className="div-table-td-title-resulttable">時間</td>
                        <td>{s2hms(summary.lengthSeconds)}</td>
                    </tr>
                    <tr>
                        <td className="div-table-td-title-resulttable">要約</td>
                        <td>{summary.concise}</td>
                    </tr>
                    <tr>
                        <td className="div-table-td-title-resulttable">キーワード</td>
                        <td>{summary.keyword.join(', ')}</td>
                    </tr>
                    <tr>
                        <td className="div-table-td-title-resulttable">トピック</td>
                        <td>
                            <ul className="div-table-ul-topic-resulttable">
                                {summary.topic.map((topic, idx) =>
                                    <li key={`title-${idx}`}>
                                        <div  className="div-table-li-topic-title-resulttable">{topic.title}</div>
                                        <ul className="div-table-ul-topic-resulttable">
                                            {topic.abstract.map((abstract, idx) =>
                                                <li key={`abstract-${idx}`}>{abstract}</li>
                                            )}
                                        </ul>
                                    </li>
                                )}
                            </ul>
                        </td>
                    </tr>
                    <tr>
                        <td className="div-table-td-title-resulttable">詳細</td>
                        <td>
                            <ul className="div-table-ul-detail-resulttable">
                                {summary.detail.map((detail, idx) =>
                                    <li key={`detail-${idx}`}>{detail}</li>
                                )}
                            </ul>
                        </td>
                    </tr>
                </tbody>
            </table>
        )
    }

    return (
        <div className="div-result">
            <ResultTable summary={summary.summary} />
        </div>
    ) 
}