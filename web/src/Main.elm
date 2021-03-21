port module Main exposing (main)

import Browser
import Device exposing (Device)
import Element exposing (Element, alignRight, centerX, centerY, fill, height, padding, paddingXY, spacing, width)
import Element.Border
import Element.Font
import FileValue as File exposing (File)
import Html exposing (Html)
import Html.Attributes
import Icon
import Json.Decode exposing (Value)
import Style


port resizes : (Device.Size -> msg) -> Sub msg


main : Program Value Model Msg
main =
    Browser.element
        { init = init
        , view = view
        , update = update
        , subscriptions = \_ -> resizes WindowResizes
        }


type alias Model =
    -- Current state of the application
    { state : State
    , device : Device
    , params : Parameters
    }


type State
    = Home FileDraggingState
    | Loading
    | Config { images : Images }
    | Processing { images : Images }
    | Results { images : Images }


type FileDraggingState
    = Idle
    | DraggingSomeFiles
    | DroppedSomeFiles File (List File)


type alias Images =
    List String


type alias Parameters =
    { crop : Maybe Crop
    , levels : Int
    , sparse : Float
    , lambda : Float
    , rho : Float
    , maxIterations : Int
    , convergenceThreshold : Float
    }


type alias Crop =
    { left : Int
    , top : Int
    , right : Int
    , bottom : Int
    }


type Msg
    = NoMsg
    | WindowResizes Device.Size
    | DragDropMsg DragDropMsg


type DragDropMsg
    = DragOver File (List File)
    | Drop File (List File)
    | DragLeave


{-| Initialize the model.
-}
init : Value -> ( Model, Cmd Msg )
init flags =
    ( { state = initialState, device = Device.default, params = defaultParams }, Cmd.none )


initialState : State
initialState =
    Home Idle


defaultParams : Parameters
defaultParams =
    { crop = Nothing
    , levels = 1
    , sparse = 0.5
    , lambda = 1.5
    , rho = 0.1
    , maxIterations = 40
    , convergenceThreshold = 0.001
    }


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case ( msg, model.state ) of
        ( NoMsg, _ ) ->
            ( model, Cmd.none )

        ( WindowResizes size, _ ) ->
            ( { model | device = Device.classify size }, Cmd.none )

        ( DragDropMsg (DragOver _ _), Home _ ) ->
            ( { model | state = Home DraggingSomeFiles }, Cmd.none )

        ( DragDropMsg (Drop file otherFiles), Home _ ) ->
            ( { model | state = Home (DroppedSomeFiles file otherFiles) }, Cmd.none )

        ( DragDropMsg DragLeave, Home _ ) ->
            ( { model | state = Home Idle }, Cmd.none )

        _ ->
            ( model, Cmd.none )



-- View ##############################################################


view : Model -> Html Msg
view model =
    Element.layout [ Style.font ]
        (viewElmUI model)


viewElmUI : Model -> Element Msg
viewElmUI model =
    case model.state of
        Home draggingState ->
            viewHome draggingState

        Loading ->
            Element.none

        Config { images } ->
            Element.none

        Processing { images } ->
            Element.none

        Results { images } ->
            Element.none


viewHome : FileDraggingState -> Element Msg
viewHome draggingState =
    Element.column (padding 20 :: width fill :: height fill :: onDropAttributes draggingState)
        [ Element.column [ centerX, spacing 16 ]
            [ Element.el [ Element.Font.size 32 ] (Element.text "Low rank image registration")
            , Element.row [ alignRight, spacing 8 ]
                [ Element.link [ Element.Font.underline ]
                    { url = "https://github.com/mpizenberg/lowrr", label = Element.text "code on GitHub" }
                , Element.el [] Element.none
                , Icon.github 16
                ]
            , Element.row [ alignRight, spacing 8 ]
                [ Element.link [ Element.Font.underline ]
                    { url = "https://hal.archives-ouvertes.fr/hal-03172399", label = Element.text "read the paper" }
                , Element.el [] Element.none
                , Icon.fileText 16
                ]
            ]
        , dropAndLoadArea draggingState
        ]


dropAndLoadArea : FileDraggingState -> Element Msg
dropAndLoadArea draggingState =
    let
        dropOrLoadText =
            Element.row []
                [ Element.text "Drop images or "
                , Element.html
                    (File.hiddenInputMultiple
                        "TheFileInput"
                        [ "image/*" ]
                        (\file otherFiles -> DragDropMsg (Drop file otherFiles))
                    )
                , Element.el [ Element.Font.underline ]
                    (Element.html
                        (Html.label [ Html.Attributes.for "TheFileInput", Html.Attributes.style "cursor" "pointer" ]
                            [ Html.text "load from disk" ]
                        )
                    )
                ]
    in
    Element.el [ width fill, height fill ]
        (case draggingState of
            Idle ->
                Element.column [ centerX, centerY, spacing 32 ]
                    [ Element.el (dropIconBorderAttributes Element.Border.dashed) (Icon.arrowDown 48)
                    , dropOrLoadText
                    ]

            DraggingSomeFiles ->
                Element.column [ centerX, centerY, spacing 32 ]
                    [ Element.el (dropIconBorderAttributes Element.Border.solid) (Icon.arrowDown 48)
                    , dropOrLoadText
                    ]

            DroppedSomeFiles _ otherFiles ->
                let
                    filesCount =
                        1 + List.length otherFiles
                in
                Element.column [ centerX, centerY, spacing 32 ]
                    [ Element.el loadingBoxBorderAttributes Element.none
                    , Element.el [ centerX ] (Element.text ("Loading " ++ String.fromInt filesCount ++ " files"))
                    ]
        )


dropIconBorderAttributes : Element.Attribute msg -> List (Element.Attribute msg)
dropIconBorderAttributes dashedAttribute =
    [ Element.Border.width 4
    , Element.Font.color Style.dropColor
    , paddingXY 16 16
    , centerX
    , dashedAttribute
    , Element.Border.rounded 16
    , width Element.shrink
    ]


loadingBoxBorderAttributes : List (Element.Attribute msg)
loadingBoxBorderAttributes =
    [ Element.Border.width 4
    , Element.Font.color Style.dropColor
    , paddingXY 0 16
    , centerX
    , Element.Border.solid
    , Element.Border.rounded 0
    , width (Element.px 400)
    ]


onDropAttributes : FileDraggingState -> List (Element.Attribute Msg)
onDropAttributes draggingState =
    case draggingState of
        DroppedSomeFiles _ _ ->
            []

        _ ->
            List.map Element.htmlAttribute
                (File.onDrop
                    { onOver = \file otherFiles -> DragDropMsg (DragOver file otherFiles)
                    , onDrop = \file otherFiles -> DragDropMsg (Drop file otherFiles)
                    , onLeave = Just { id = "FileDropArea", msg = DragDropMsg DragLeave }
                    }
                )
