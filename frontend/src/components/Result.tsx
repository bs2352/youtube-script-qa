import { SummaryResponseBody } from "./types"
import './Result.css'

interface Props {
    summary: SummaryResponseBody
}

export function Result ({
    summary,
}: Props) {

    const ResultTable = ({
        summary
    } : Props) : JSX.Element => {
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
                            <ul style={{listStyle: "none"}} >
                                {summary.topic.map((topic, idx) =>
                                    <li key={`title-${idx}`}>
                                        <div  className="div-table-li-topic-title-resulttable">{topic.title}</div>
                                        <ul style={{listStyle: "none"}}>
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
                            <ul>
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
            <ResultTable summary={summary} />
        </div>
    ) 
}