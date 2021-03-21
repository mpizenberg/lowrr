module Icon exposing (arrowDown, fileText, github)

import Element exposing (Element)
import FeatherIcons



-- import Html exposing (Html)
-- import Svg exposing (Svg, svg)
-- import Svg.Attributes exposing (..)


featherIcon : FeatherIcons.Icon -> Float -> Element msg
featherIcon icon size =
    Element.html (FeatherIcons.toHtml [] (FeatherIcons.withSize size icon))


github : Float -> Element msg
github =
    featherIcon FeatherIcons.github


fileText : Float -> Element msg
fileText =
    featherIcon FeatherIcons.fileText


arrowDown : Float -> Element msg
arrowDown =
    featherIcon FeatherIcons.arrowDown
