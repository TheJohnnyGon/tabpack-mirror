PRAGMA yt.InferSchema = '1';
DECLARE $input1 AS String;
DECLARE $input2 AS String;
DECLARE $output1 AS String;

-- $input1: DocId, RawFormulaVal        (предсказания модели)
-- $input2: DocId, GroupId, Label       (метаданные документов)
--
-- Шаг 1: JOIN предсказаний с метаданными по DocId.
-- Шаг 2: для каждой GroupId оставляем топ-10 по RawFormulaVal DESC.

$joined = (
    SELECT
        preds.DocId AS DocId,
        meta.Label AS Label,
        CAST(preds.RawFormulaVal AS Double) AS RawFormulaVal,
        meta.GroupId AS GroupId
    FROM $input1 AS preds
    JOIN $input2 AS meta
        ON preds.DocId == meta.DocId
    WHERE meta.Label != "Label"
);

INSERT INTO $output1 WITH TRUNCATE
SELECT DocId, Label, RawFormulaVal, GroupId
FROM (
    SELECT
        DocId, Label, RawFormulaVal, GroupId,
        ROW_NUMBER() OVER (
            PARTITION BY GroupId
            ORDER BY RawFormulaVal DESC
        ) AS rn
    FROM $joined
) WHERE rn <= 10;
