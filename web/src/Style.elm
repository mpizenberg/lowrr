module Style exposing (black, dropColor, font, green, lightGrey, white)

import Element
import Element.Font as Font


font : Element.Attribute msg
font =
    Font.family
        [ Font.typeface "Open Sans"
        , Font.sansSerif
        ]



-- Color


dropColor : Element.Color
dropColor =
    Element.rgb255 50 50 250


lightGrey : Element.Color
lightGrey =
    Element.rgb255 187 187 187


darkGrey : Element.Color
darkGrey =
    Element.rgb255 50 50 50


green : Element.Color
green =
    Element.rgb255 39 203 139


red : Element.Color
red =
    Element.rgb255 203 60 60


darkRed : Element.Color
darkRed =
    Element.rgb255 70 20 20


white : Element.Color
white =
    Element.rgb255 255 255 255


black : Element.Color
black =
    Element.rgb255 0 0 0
