module Style exposing
    ( font
    , dropColor, errorColor
    , black, white, almostWhite, lightGrey, green
    )

{-| Style of our application

@docs font

@docs dropColor, errorColor

@docs black, white, almostWhite, lightGrey, green

-}

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


errorColor : Element.Color
errorColor =
    Element.rgb255 150 50 50


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


almostWhite : Element.Color
almostWhite =
    Element.rgb255 235 235 235


black : Element.Color
black =
    Element.rgb255 0 0 0
