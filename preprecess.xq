for $text in //text
  let $text-words := 
    for $w in $text//w
      where $w/@lemma != '|'
      return
        <w lemma="{fn:translate($w/@lemma, '|', '')}">
          {data($w)}
        </w>
  return
    <text party="{$text/@party}" year="{$text/@year}">
      {$text-words}
    </text>