PRAGMA yt.HybridDqExecution="1";
$INPUT_TABLE_BASKET = {{input1->table_quote()}};
$INPUT_TABLE_AFTER_SOY = {{input2->table_quote()}};
$OUTPUT_TABLE = {{output1->table_quote()}};
$FLAGS_SETS = {{param.flags_sets->quote()}};
$COMPARE_WITH_AA = {{param.compare_with_aa->quote()}};
$AA_FLAGS_SET = {{param.aa_flags_set->quote()}};


PRAGMA AnsiInForEmptyOrNullableItemsCollections;
PRAGMA yt.DefaultMaxJobFails = "1";
PRAGMA yson.DisableStrict;
PRAGMA yt.UseNativeYtTypes;
PRAGMA library("replay_common.sql");
IMPORT replay_common SYMBOLS $makeId, $fsToString;
PRAGMA library("replay_common_py.sql");
IMPORT replay_common_py SYMBOLS $parseFlagsSets;

$compareWithAa = Unwrap(CAST(String::AsciiToLower($COMPARE_WITH_AA) AS Bool));
$flagsSetsParsed = $parseFlagsSets($FLAGS_SETS);
$aaFlagsSet = Unwrap($parseFlagsSets("[" || $AA_FLAGS_SET || "]")[0]);
$flagsSets = IF(
    $compareWithAa AND $aaFlagsSet NOT IN $flagsSetsParsed,
    ListExtend($flagsSetsParsed, [$aaFlagsSet]),
    $flagsSetsParsed
);
$flagsSets = ListMap(
    $flagsSets, $fsToString
);

$SEP = "$_$";

$afterSoy_ = (
    SELECT
        Unwrap(String::SplitToList(`id`, $SEP)[0]) AS user_id,
        Unwrap(String::SplitToList(`id`, $SEP)[1]) AS session_id,
        Unwrap(String::SplitToList(`id`, $SEP)[2]) AS table_type,
        Unwrap(String::SplitToList(`id`, $SEP)[3]) AS country,
        Unwrap(String::SplitToList(`id`, $SEP)[4]) AS fielddate,
        Unwrap(String::SplitToList(`id`, $SEP)[5]) AS right_answer_dedup_id,
        CAST(Unwrap(String::SplitToList(`id`, $SEP)[6]) AS Int64) AS q_num,
        Unwrap(String::SplitToList(`id`, $SEP)[7]) AS fs_to_string,
        ListMap(answers_and_click.Results, $makeId) AS test_results,
    FROM $INPUT_TABLE_AFTER_SOY
);

$ensureUnique = ($list) -> {
    RETURN DictItems(ToDict($list));
};

DEFINE SUBQUERY $afterSoy() AS
    SELECT
        user_id,
        session_id,
        table_type,
        country,
        fielddate,
        right_answer_dedup_id,
        fs_to_string,
        ListMap(
            ListSort($ensureUnique(AGGREGATE_LIST(AsTuple(q_num, test_results)))), ($x)->($x.1)
        ) AS test_results
    FROM $afterSoy_
    GROUP BY 
        user_id,
        session_id,
        table_type,
        country,
        fielddate,
        right_answer_dedup_id,
        fs_to_string;
END DEFINE;

DEFINE SUBQUERY $processBeforeSoyNull($bsTable, $_) AS
    SELECT * FROM $bsTable;
END DEFINE;

DEFINE SUBQUERY $processBeforeSoyReplaceAA($bsTable, $as) AS
    $aa = (
        SELECT
            *
        FROM $as()
        WHERE fs_to_string = $fsToString($aaFlagsSet));

    $patchResultsAA = ($results, $testResults, $rightList) -> {
        RETURN ListMap(
            ListZip($results, $testResults),
            ($tup)->(
                RenameMembers(RemoveMembers(
                    ExpandStruct(
                    $tup.0,
                    ($tup.0).results_prod AS results_prod_logs,
                    ($tup.0).right_answer_index_prod AS right_answer_index_prod_logs,
                    $tup.1 AS results_prod_,
                    ListMin(ListNotNull(ListMap(
                        $rightList,
                        ($right)->(ListIndexOf($tup.1, $right))
                    ))) AS right_answer_index_prod_
                ),
                ["results_prod", "right_answer_index_prod"]), [("results_prod_", "results_prod"), ("right_answer_index_prod_", "right_answer_index_prod")])
            )
        )
    };

    SELECT
        $patchResultsAA(b.requests, test_results, b.rids) AS requests,
        b.*
    WITHOUT b.requests
    FROM $bsTable AS b
    INNER JOIN ANY $aa AS a USING (
        user_id,
        session_id,
        table_type,
        country,
        fielddate,
        right_answer_dedup_id);
END DEFINE;

$multiplyInputSubquery = EvaluateCode(IF(
    $compareWithAa,
    QuoteCode($processBeforeSoyReplaceAA),
    QuoteCode($processBeforeSoyNull)
));

$beforeSoyProcessed = PROCESS $multiplyInputSubquery($INPUT_TABLE_BASKET, $afterSoy);

$patchResults = ($results, $testResults, $rightList) -> {
    RETURN ListMap(
        ListZip($results, $testResults),
        ($tup)->(
            ExpandStruct(
                $tup.0,
                Unwrap($tup.1) AS results_test,
                ListMin(ListNotNull(ListMap(
                    $rightList,
                    ($right)->(ListIndexOf(Unwrap($tup.1), $right))
                ))) AS right_answer_index_test
            )
        )
    )
};

$beforeSoy = (
    SELECT
        *
    FROM (
        SELECT
            t.*,
            $flagsSets AS fs_to_string
        FROM $beforeSoyProcessed AS t)
    FLATTEN LIST BY fs_to_string);

$join = (
    SELECT
        $patchResults(b.requests, test_results, b.rids) AS requests,
        b.* WITHOUT b.requests
    FROM $beforeSoy AS b
    INNER JOIN ANY $afterSoy() AS a
    USING (
        user_id,
        session_id,
        table_type,
        country,
        fielddate,
        right_answer_dedup_id,
        fs_to_string));

INSERT INTO
    $OUTPUT_TABLE
WITH TRUNCATE 
SELECT
    *
FROM $join
ORDER BY
    user_id,
    session_id,
    table_type,
    country,
    fielddate,
    right_answer_dedup_id,
    fs_to_string;
