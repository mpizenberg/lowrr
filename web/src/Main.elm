port module Main exposing (main)

import Browser
import Device exposing (Device)
import Dict exposing (Dict)
import Element exposing (Element, alignRight, centerX, centerY, fill, height, padding, paddingXY, spacing, width)
import Element.Background
import Element.Border
import Element.Font
import FileValue as File exposing (File)
import Html exposing (Html)
import Html.Attributes
import Icon
import Json.Decode exposing (Value)
import Keyboard exposing (RawKey)
import Pivot exposing (Pivot)
import Set exposing (Set)
import Simple.Transition as Transition
import Style
import Svg
import Svg.Attributes
import Viewer
import Viewer.Svg


port resizes : (Device.Size -> msg) -> Sub msg


port decodeImages : List Value -> Cmd msg


port imageDecoded : (Image -> msg) -> Sub msg


main : Program Device.Size Model Msg
main =
    Browser.element
        { init = init
        , view = view
        , update = update
        , subscriptions = subscriptions
        }


type alias Model =
    -- Current state of the application
    { state : State
    , device : Device
    , params : Parameters
    }


type State
    = Home FileDraggingState
    | Loading { names : Set String, loaded : Dict String Image }
    | ViewImgs { images : Pivot Image }
    | Config { images : Pivot Image }
    | Processing { images : Images }
    | Results { images : Images }


type FileDraggingState
    = Idle
    | DraggingSomeFiles


type alias Image =
    { id : String
    , url : String
    , width : Int
    , height : Int
    }


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


{-| Initialize the model.
-}
init : Device.Size -> ( Model, Cmd Msg )
init size =
    ( { state = initialState, device = Device.classify size, params = defaultParams }, Cmd.none )


initialState : State
initialState =
    -- Home Idle
    -- ViewImgs { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }
    Config { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }


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



-- Update ############################################################


type Msg
    = NoMsg
    | WindowResizes Device.Size
    | DragDropMsg DragDropMsg
    | ImageDecoded Image
    | KeyDown RawKey


type DragDropMsg
    = DragOver File (List File)
    | Drop File (List File)
    | DragLeave


subscriptions : Model -> Sub Msg
subscriptions model =
    case model.state of
        Home _ ->
            Sub.batch [ resizes WindowResizes, imageDecoded ImageDecoded ]

        Loading _ ->
            Sub.batch [ resizes WindowResizes, imageDecoded ImageDecoded ]

        ViewImgs _ ->
            Sub.batch [ resizes WindowResizes, Keyboard.downs KeyDown ]

        Config _ ->
            Sub.batch [ resizes WindowResizes ]

        Processing _ ->
            Sub.batch [ resizes WindowResizes ]

        Results _ ->
            Sub.batch [ resizes WindowResizes ]


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
            let
                imageFiles =
                    List.filter (\f -> String.startsWith "image" f.mime) (file :: otherFiles)

                names =
                    Set.fromList (List.map .name imageFiles)
            in
            ( { model | state = Loading { names = names, loaded = Dict.empty } }
            , decodeImages (List.map File.encode imageFiles)
            )

        ( DragDropMsg DragLeave, Home _ ) ->
            ( { model | state = Home Idle }, Cmd.none )

        ( ImageDecoded img, Loading { names, loaded } ) ->
            let
                updatedLoadingState =
                    { names = names
                    , loaded = Dict.insert img.id img loaded
                    }
            in
            if Set.size names == Dict.size updatedLoadingState.loaded then
                case Dict.values updatedLoadingState.loaded of
                    [] ->
                        -- This should be impossible, there must be at least 1 image
                        ( { model | state = Home Idle }, Cmd.none )

                    firstImage :: otherImages ->
                        ( { model | state = ViewImgs { images = Pivot.fromCons firstImage otherImages } }
                        , Cmd.none
                        )

            else
                ( { model | state = Loading updatedLoadingState }, Cmd.none )

        ( KeyDown rawKey, ViewImgs { images } ) ->
            case Keyboard.navigationKey rawKey of
                Just Keyboard.ArrowRight ->
                    ( { model | state = ViewImgs { images = Pivot.goR images |> Maybe.withDefault (Pivot.goToStart images) } }
                    , Cmd.none
                    )

                Just Keyboard.ArrowLeft ->
                    ( { model | state = ViewImgs { images = Pivot.goL images |> Maybe.withDefault (Pivot.goToEnd images) } }
                    , Cmd.none
                    )

                _ ->
                    ( model, Cmd.none )

        _ ->
            ( model, Cmd.none )



-- View ##############################################################


view : Model -> Html Msg
view model =
    Element.layout [ Style.font, Element.clip ]
        (viewElmUI model)


viewElmUI : Model -> Element Msg
viewElmUI model =
    case model.state of
        Home draggingState ->
            viewHome draggingState

        Loading loadData ->
            viewLoading loadData

        ViewImgs { images } ->
            viewImgs images model.device

        Config { images } ->
            viewConfig images model.params model.device

        Processing { images } ->
            Element.none

        Results { images } ->
            Element.none


viewConfig : Pivot Image -> Parameters -> Device -> Element Msg
viewConfig images params device =
    Debug.todo "config"


viewImgs : Pivot Image -> Device -> Element Msg
viewImgs images device =
    let
        img =
            Pivot.getC images

        imgSvgAttributes =
            [ Svg.Attributes.xlinkHref img.url
            , Svg.Attributes.width (String.fromInt img.width)
            , Svg.Attributes.height (String.fromInt img.height)
            ]

        viewerAttributes =
            Viewer.withSize ( device.size.width, device.size.height )
                |> Viewer.fitImage 1.0 ( toFloat img.width, toFloat img.height )
                |> Viewer.Svg.transform
    in
    Element.html <|
        Svg.svg
            [ Html.Attributes.width (floor device.size.width)
            , Html.Attributes.height (floor device.size.height)
            ]
            [ Svg.g [ viewerAttributes ] [ Svg.image imgSvgAttributes [] ] ]


viewHome : FileDraggingState -> Element Msg
viewHome draggingState =
    Element.column (padding 20 :: width fill :: height fill :: onDropAttributes)
        [ viewTitle
        , dropAndLoadArea draggingState
        ]


viewLoading : { names : Set String, loaded : Dict String Image } -> Element Msg
viewLoading { names, loaded } =
    let
        totalCount =
            Set.size names

        loadCount =
            Dict.size loaded
    in
    Element.column [ padding 20, width fill, height fill ]
        [ viewTitle
        , Element.el [ width fill, height fill ]
            (Element.column
                [ centerX, centerY, spacing 32 ]
                [ Element.el loadingBoxBorderAttributes (loadBar loadCount totalCount)
                , Element.el [ centerX ] (Element.text ("Loading " ++ String.fromInt totalCount ++ " images"))
                ]
            )
        ]


loadBar : Int -> Int -> Element msg
loadBar loaded total =
    let
        barLength =
            (400 - 2 * 4) * loaded // total
    in
    Element.el
        [ width (Element.px barLength)
        , height Element.fill
        , Element.Background.color Style.dropColor
        , Element.htmlAttribute
            (Transition.properties
                [ Transition.property "width" 200 [] ]
            )
        ]
        Element.none


viewTitle : Element msg
viewTitle =
    Element.column [ centerX, spacing 16 ]
        [ Element.paragraph [ Element.Font.center, Element.Font.size 32 ] [ Element.text "Low rank image registration" ]
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


dropAndLoadArea : FileDraggingState -> Element Msg
dropAndLoadArea draggingState =
    let
        borderStyle =
            case draggingState of
                Idle ->
                    Element.Border.dashed

                DraggingSomeFiles ->
                    Element.Border.solid

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
        (Element.column [ centerX, centerY, spacing 32 ]
            [ Element.el (dropIconBorderAttributes borderStyle) (Icon.arrowDown 48)
            , dropOrLoadText
            ]
        )


dropIconBorderAttributes : Element.Attribute msg -> List (Element.Attribute msg)
dropIconBorderAttributes dashedAttribute =
    [ Element.Border.width 4
    , Element.Font.color Style.dropColor
    , centerX
    , Element.clip

    -- below is different
    , paddingXY 16 16
    , dashedAttribute
    , Element.Border.rounded 16
    , height (Element.px (48 + (16 + 4) * 2))
    , width (Element.px (48 + (16 + 4) * 2))
    , borderTransition
    ]


loadingBoxBorderAttributes : List (Element.Attribute msg)
loadingBoxBorderAttributes =
    [ Element.Border.width 4
    , Element.Font.color Style.dropColor
    , centerX
    , Element.clip

    -- below is different
    , paddingXY 0 0
    , Element.Border.solid
    , Element.Border.rounded 0
    , height (Element.px ((16 + 4) * 2))
    , width (Element.px 325)
    , borderTransition
    ]


borderTransition : Element.Attribute msg
borderTransition =
    Element.htmlAttribute
        (Transition.properties
            [ Transition.property "border-radius" 300 []
            , Transition.property "height" 300 []
            , Transition.property "width" 300 []
            ]
        )


onDropAttributes : List (Element.Attribute Msg)
onDropAttributes =
    List.map Element.htmlAttribute
        (File.onDrop
            { onOver = \file otherFiles -> DragDropMsg (DragOver file otherFiles)
            , onDrop = \file otherFiles -> DragDropMsg (Drop file otherFiles)
            , onLeave = Just { id = "FileDropArea", msg = DragDropMsg DragLeave }
            }
        )
