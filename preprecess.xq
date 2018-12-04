
let $corpus :=
  for $text in //text
  for $sent in $text/sentence
    let $sent-words :=
      for $w in $sent//w
        return
          <w lemma="{fn:translate($w/@lemma, '|', '')}">
            {data($w)}
          </w>
    return
      <text party="{$text/@party}" year="{$text/@year}" id="{fn:concat($text/@party, '-', $text/@type, '-', $text/@year)}">
        {$sent-words}
      </text>
    
return
  <corpus>
    {$corpus}
  </corpus>