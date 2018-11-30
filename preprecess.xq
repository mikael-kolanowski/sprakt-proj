
let $corpus :=
  for $text in //text
    let $text-words := 
      for $w in $text//w
        return
          <w lemma="{fn:translate($w/@lemma, '|', '')}">
            {data($w)}
          </w>
    return
      <text party="{$text/@party}" year="{$text/@year}" id="{fn:concat($text/@party, '-', $text/@type, '-', $text/@year)}">
        {$text-words}
      </text>
    
return
  <corpus>
    {$corpus}
  </corpus>