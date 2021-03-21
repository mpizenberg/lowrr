module Style exposing (dropColor, font)

import Element
import Element.Font as Font


font : Element.Attribute msg
font =
    Font.family
        [ Font.typeface "Open Sans"
        , Font.sansSerif
        ]


dropColor : Element.Color
dropColor =
    Element.rgb255 50 50 250
