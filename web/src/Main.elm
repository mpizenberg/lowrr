port module Main exposing (main)

import Browser
import Device exposing (Device)
import Dict exposing (Dict)
import Element exposing (Element, alignRight, centerX, centerY, fill, height, padding, paddingXY, spacing, width)
import Element.Background
import Element.Border
import Element.Font
import Element.Input
import FileValue as File exposing (File)
import Html exposing (Html)
import Html.Attributes
import Icon
import Json.Decode exposing (Value)
import Keyboard exposing (RawKey)
import NumberInput
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
    , paramsForm : ParametersForm
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
    , equalize : Bool
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


type alias ParametersForm =
    { crop : CropForm
    , maxIterations : NumberInput.Field Int NumberInput.IntError
    , convergenceThreshold : NumberInput.Field Float NumberInput.FloatError
    , levels : NumberInput.Field Int NumberInput.IntError
    , sparse : NumberInput.Field Float NumberInput.FloatError
    , lambda : NumberInput.Field Float NumberInput.FloatError
    , rho : NumberInput.Field Float NumberInput.FloatError
    }


type alias CropForm =
    { active : Bool
    , left : NumberInput.Field Int NumberInput.IntError
    , top : NumberInput.Field Int NumberInput.IntError
    , right : NumberInput.Field Int NumberInput.IntError
    , bottom : NumberInput.Field Int NumberInput.IntError
    }


{-| Initialize the model.
-}
init : Device.Size -> ( Model, Cmd Msg )
init size =
    ( { state = initialState
      , device = Device.classify size
      , params = defaultParams
      , paramsForm = defaultParamsForm
      }
    , Cmd.none
    )


initialState : State
initialState =
    -- Home Idle
    -- ViewImgs { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }
    Config { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }


defaultParams : Parameters
defaultParams =
    { crop = Nothing
    , equalize = True
    , levels = 4
    , sparse = 0.5
    , lambda = 1.5
    , rho = 0.1
    , maxIterations = 40
    , convergenceThreshold = 0.001
    }


defaultParamsForm : ParametersForm
defaultParamsForm =
    let
        anyInt =
            NumberInput.intDefault

        anyFloat =
            NumberInput.floatDefault
    in
    { crop = defaultCropForm 1920 1080
    , maxIterations =
        { anyInt | min = Just 1, max = Just 1000 }
            |> NumberInput.setDefaultIntValue defaultParams.maxIterations
    , convergenceThreshold =
        { defaultValue = defaultParams.convergenceThreshold
        , min = Just 0.0
        , max = Nothing
        , increase = \x -> x * sqrt 2
        , decrease = \x -> x / sqrt 2
        , input = String.fromFloat defaultParams.convergenceThreshold
        , decodedInput = Ok defaultParams.convergenceThreshold
        }
    , levels =
        { anyInt | min = Just 1, max = Just 10 }
            |> NumberInput.setDefaultIntValue defaultParams.levels
    , sparse =
        { anyFloat | min = Just 0.0, max = Just 1.0 }
            |> NumberInput.setDefaultFloatValue defaultParams.sparse
    , lambda =
        { anyFloat | min = Just 0.0 }
            |> NumberInput.setDefaultFloatValue defaultParams.lambda
    , rho =
        { defaultValue = defaultParams.rho
        , min = Just 0.0
        , max = Nothing
        , increase = \x -> x * sqrt 2
        , decrease = \x -> x / sqrt 2
        , input = String.fromFloat defaultParams.rho
        , decodedInput = Ok defaultParams.rho
        }
    }


defaultCropForm : Int -> Int -> CropForm
defaultCropForm width height =
    let
        anyInt =
            NumberInput.intDefault
    in
    { active = defaultParams.crop /= Nothing
    , left =
        { anyInt | min = Just 0, max = Just width }
            |> NumberInput.setDefaultIntValue 0
    , top =
        { anyInt | min = Just 0, max = Just height }
            |> NumberInput.setDefaultIntValue 0
    , right =
        { anyInt | min = Just 0, max = Just width }
            |> NumberInput.setDefaultIntValue width
    , bottom =
        { anyInt | min = Just 0, max = Just height }
            |> NumberInput.setDefaultIntValue height
    }



-- Update ############################################################


type Msg
    = NoMsg
    | WindowResizes Device.Size
    | DragDropMsg DragDropMsg
    | ImageDecoded Image
    | KeyDown RawKey
    | ParamsMsg ParamsMsg


type DragDropMsg
    = DragOver File (List File)
    | Drop File (List File)
    | DragLeave


type ParamsMsg
    = ToggleEqualize Bool
    | ChangeMaxIter String
    | ChangeConvergenceThreshold String
    | ChangeLevels String
    | ChangeSparse String
    | ChangeLambda String
    | ChangeRho String
    | ToggleCrop Bool
    | ChangeCropLeft String
    | ChangeCropTop String
    | ChangeCropRight String
    | ChangeCropBottom String


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

        ( ParamsMsg paramsMsg, Config _ ) ->
            let
                ( newParams, newParamsForm ) =
                    updateParams paramsMsg ( model.params, model.paramsForm )
            in
            ( { model | params = newParams, paramsForm = newParamsForm }, Cmd.none )

        _ ->
            ( model, Cmd.none )


updateParams : ParamsMsg -> ( Parameters, ParametersForm ) -> ( Parameters, ParametersForm )
updateParams msg ( params, paramsForm ) =
    case msg of
        ToggleEqualize equalize ->
            ( { params | equalize = equalize }, paramsForm )

        ChangeMaxIter str ->
            let
                updatedField =
                    NumberInput.updateInt str paramsForm.maxIterations

                updatedForm =
                    { paramsForm | maxIterations = updatedField }
            in
            case updatedField.decodedInput of
                Ok maxIterations ->
                    ( { params | maxIterations = maxIterations }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeConvergenceThreshold str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.convergenceThreshold

                updatedForm =
                    { paramsForm | convergenceThreshold = updatedField }
            in
            case updatedField.decodedInput of
                Ok convergenceThreshold ->
                    ( { params | convergenceThreshold = convergenceThreshold }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeLevels str ->
            let
                updatedField =
                    NumberInput.updateInt str paramsForm.levels

                updatedForm =
                    { paramsForm | levels = updatedField }
            in
            case updatedField.decodedInput of
                Ok levels ->
                    ( { params | levels = levels }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeSparse str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.sparse

                updatedForm =
                    { paramsForm | sparse = updatedField }
            in
            case updatedField.decodedInput of
                Ok sparse ->
                    ( { params | sparse = sparse }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeLambda str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.lambda

                updatedForm =
                    { paramsForm | lambda = updatedField }
            in
            case updatedField.decodedInput of
                Ok lambda ->
                    ( { params | lambda = lambda }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeRho str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.rho

                updatedForm =
                    { paramsForm | rho = updatedField }
            in
            case updatedField.decodedInput of
                Ok rho ->
                    ( { params | rho = rho }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ToggleCrop activeCrop ->
            let
                oldCropForm =
                    paramsForm.crop

                newCropForm =
                    { oldCropForm | active = activeCrop }
            in
            case ( activeCrop, ( newCropForm.left.decodedInput, newCropForm.right.decodedInput ), ( newCropForm.top.decodedInput, newCropForm.bottom.decodedInput ) ) of
                ( True, ( Ok left, Ok right ), ( Ok top, Ok bottom ) ) ->
                    ( { params | crop = Just (Crop left top right bottom) }
                    , { paramsForm | crop = newCropForm }
                    )

                _ ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = newCropForm }
                    )

        ChangeCropLeft str ->
            let
                oldCropForm =
                    paramsForm.crop

                newLeft =
                    NumberInput.updateInt str oldCropForm.left
            in
            case ( oldCropForm.active, newLeft.decodedInput ) of
                ( True, Ok left ) ->
                    let
                        newRight =
                            NumberInput.setMinBound (Just left) oldCropForm.right
                                |> NumberInput.updateInt oldCropForm.right.input

                        newCropForm =
                            { oldCropForm | left = newLeft, right = newRight }

                        newCrop =
                            case ( newRight.decodedInput, oldCropForm.top.decodedInput, oldCropForm.bottom.decodedInput ) of
                                ( Ok right, Ok top, Ok bottom ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | left = newLeft } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )

        ChangeCropTop str ->
            let
                oldCropForm =
                    paramsForm.crop

                newTop =
                    NumberInput.updateInt str oldCropForm.top
            in
            case ( oldCropForm.active, newTop.decodedInput ) of
                ( True, Ok top ) ->
                    let
                        newBottom =
                            NumberInput.setMinBound (Just top) oldCropForm.bottom
                                |> NumberInput.updateInt oldCropForm.bottom.input

                        newCropForm =
                            { oldCropForm | top = newTop, bottom = newBottom }

                        newCrop =
                            case ( newBottom.decodedInput, oldCropForm.left.decodedInput, oldCropForm.right.decodedInput ) of
                                ( Ok bottom, Ok left, Ok right ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | top = newTop } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )

        ChangeCropRight str ->
            let
                oldCropForm =
                    paramsForm.crop

                newRight =
                    NumberInput.updateInt str oldCropForm.right
            in
            case ( oldCropForm.active, newRight.decodedInput ) of
                ( True, Ok right ) ->
                    let
                        newCropForm =
                            { oldCropForm | right = newRight }

                        newCrop =
                            case ( oldCropForm.left.decodedInput, oldCropForm.top.decodedInput, oldCropForm.bottom.decodedInput ) of
                                ( Ok left, Ok top, Ok bottom ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | right = newRight } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )

        ChangeCropBottom str ->
            let
                oldCropForm =
                    paramsForm.crop

                newBottom =
                    NumberInput.updateInt str oldCropForm.bottom
            in
            case ( oldCropForm.active, newBottom.decodedInput ) of
                ( True, Ok bottom ) ->
                    let
                        newCropForm =
                            { oldCropForm | bottom = newBottom }

                        newCrop =
                            case ( oldCropForm.left.decodedInput, oldCropForm.top.decodedInput, oldCropForm.right.decodedInput ) of
                                ( Ok left, Ok top, Ok right ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | bottom = newBottom } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )



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
            viewConfig images model.params model.paramsForm model.device

        Processing { images } ->
            Element.none

        Results { images } ->
            Element.none


viewConfig : Pivot Image -> Parameters -> ParametersForm -> Device -> Element Msg
viewConfig images params paramsForm device =
    Element.column [ padding 20, spacing 32 ]
        [ Element.el [ Element.Font.center, Element.Font.size 32 ] (Element.text "Algorithm parameters")

        -- Cropped working frame
        , Element.column [ spacing 10 ]
            [ Element.text "Cropped working frame:"
            , Element.row [ spacing 10 ]
                [ Element.text "off"
                , toggle (ParamsMsg << ToggleCrop) paramsForm.crop.active 30 "Toggle cropped working frame"
                , Element.text "on"
                ]
            , cropBox paramsForm.crop
            ]

        -- Equalize mean intensities
        , Element.column [ spacing 10 ]
            [ Element.text "Equalize mean intensities:"
            , Element.row [ spacing 10 ]
                [ Element.text "off"
                , toggle (ParamsMsg << ToggleEqualize) params.equalize 30 "Toggle mean intensities equalization"
                , Element.text "on"
                ]
            ]

        -- Maximum number of iterations
        , Element.column [ spacing 10 ]
            [ Element.text "Maximum number of iterations:"
            , Element.text ("(default to " ++ String.fromInt defaultParams.maxIterations ++ ")")
            , intInput paramsForm.maxIterations (ParamsMsg << ChangeMaxIter) "Maximum number of iterations"
            , displayIntErrors paramsForm.maxIterations.decodedInput
            ]

        -- Convergence threshold
        , Element.column [ spacing 10 ]
            [ Element.text "Convergence threshold:"
            , Element.text ("(default to " ++ String.fromFloat defaultParams.convergenceThreshold ++ ")")
            , floatInput paramsForm.convergenceThreshold (ParamsMsg << ChangeConvergenceThreshold) "Convergence threshold"
            , displayFloatErrors paramsForm.convergenceThreshold.decodedInput
            ]

        -- Multi-resolution pyramid levels
        , Element.column [ spacing 10 ]
            [ Element.text "Number of pyramid levels:"
            , Element.text ("(default to " ++ String.fromInt defaultParams.levels ++ ")")
            , intInput paramsForm.levels (ParamsMsg << ChangeLevels) "Number of pyramid levels"
            , displayIntErrors paramsForm.levels.decodedInput
            ]

        -- Sparse ratio threshold
        , Element.column [ spacing 10 ]
            [ Element.text "Sparse ratio threshold to switch:"
            , Element.text ("(default to " ++ String.fromFloat defaultParams.sparse ++ ")")
            , floatInput paramsForm.sparse (ParamsMsg << ChangeSparse) "Sparse ratio threshold to switch"
            , displayFloatErrors paramsForm.sparse.decodedInput
            ]

        -- lambda
        , Element.column [ spacing 10 ]
            [ Element.text ("lambda: (default to " ++ String.fromFloat defaultParams.lambda ++ ")")
            , floatInput paramsForm.lambda (ParamsMsg << ChangeLambda) "lambda"
            , displayFloatErrors paramsForm.lambda.decodedInput
            ]

        -- rho
        , Element.column [ spacing 10 ]
            [ Element.text ("rho: (default to " ++ String.fromFloat defaultParams.rho ++ ")")
            , floatInput paramsForm.rho (ParamsMsg << ChangeRho) "rho"
            , displayFloatErrors paramsForm.rho.decodedInput
            ]
        ]



-- Crop input


cropBox : CropForm -> Element Msg
cropBox cropForm =
    if not cropForm.active then
        Element.none

    else
        Element.el [ width fill, padding 4 ] <|
            Element.el
                [ centerX
                , centerY
                , paddingXY 48 20
                , Element.Border.dashed
                , Element.Border.width 2
                , Element.onLeft
                    (Element.el (Element.moveRight 30 :: onBorderAttributes)
                        (cropField "left" (ParamsMsg << ChangeCropLeft) cropForm.left)
                    )
                , Element.above
                    (Element.el (Element.moveDown 12 :: onBorderAttributes)
                        (cropField "top" (ParamsMsg << ChangeCropTop) cropForm.top)
                    )
                , Element.onRight
                    (Element.el (Element.moveLeft 30 :: onBorderAttributes)
                        (cropField "right" (ParamsMsg << ChangeCropRight) cropForm.right)
                    )
                , Element.below
                    (Element.el (Element.moveUp 14 :: onBorderAttributes)
                        (cropField "bottom" (ParamsMsg << ChangeCropBottom) cropForm.bottom)
                    )
                ]
                (Element.el [ Element.Font.size 12 ] <|
                    case ( decodedCropWidth cropForm, decodedCropHeight cropForm ) of
                        ( Just cropWidth, Just cropHeight ) ->
                            Element.text (String.fromInt cropWidth ++ " x " ++ String.fromInt cropHeight)

                        ( Nothing, Just cropHeight ) ->
                            Element.text ("? x " ++ String.fromInt cropHeight)

                        ( Just cropWidth, Nothing ) ->
                            Element.text (String.fromInt cropWidth ++ " x ?")

                        ( Nothing, Nothing ) ->
                            Element.text "? x ?"
                )


decodedCropWidth : CropForm -> Maybe Int
decodedCropWidth cropForm =
    case ( cropForm.left.decodedInput, cropForm.right.decodedInput ) of
        ( Ok left, Ok right ) ->
            Just (right - left)

        _ ->
            Nothing


decodedCropHeight : CropForm -> Maybe Int
decodedCropHeight cropForm =
    case ( cropForm.top.decodedInput, cropForm.bottom.decodedInput ) of
        ( Ok top, Ok bottom ) ->
            Just (bottom - top)

        _ ->
            Nothing


onBorderAttributes : List (Element.Attribute msg)
onBorderAttributes =
    [ centerX, centerY, Element.Background.color Style.white ]


cropField : String -> (String -> msg) -> NumberInput.Field Int NumberInput.IntError -> Element msg
cropField label msgTag field =
    Element.Input.text [ paddingXY 0 4, Element.Border.width 0, Element.Font.center, width (Element.px 60) ]
        { onChange = msgTag
        , text = field.input
        , placeholder = Nothing
        , label = Element.Input.labelHidden label
        }



-- Int input


displayIntErrors : Result (List NumberInput.IntError) a -> Element msg
displayIntErrors result =
    case result of
        Ok _ ->
            Element.none

        Err errors ->
            List.map Debug.toString errors
                |> String.join ", "
                |> Element.text


intInput : NumberInput.Field Int NumberInput.IntError -> (String -> msg) -> String -> Element msg
intInput field msgTag label =
    let
        textField =
            Element.Input.text [ Element.Border.width 0, Element.Font.center, width (Element.px 100) ]
                { onChange = msgTag
                , text = field.input
                , placeholder = Nothing
                , label = Element.Input.labelHidden label
                }
    in
    case field.decodedInput of
        Err _ ->
            Element.row [ Element.Border.solid, Element.Border.width 1, Element.Border.rounded 4 ]
                [ numberSideButton Nothing "−"
                , textField
                , numberSideButton Nothing "+"
                ]

        Ok current ->
            let
                increased =
                    field.increase current

                decreased =
                    field.decrease current

                decrementMsg =
                    case field.min of
                        Nothing ->
                            Just (msgTag (String.fromInt decreased))

                        Just minBound ->
                            if current <= minBound then
                                Nothing

                            else
                                Just (msgTag (String.fromInt <| max decreased minBound))

                incrementMsg =
                    case field.max of
                        Nothing ->
                            Just (msgTag (String.fromInt increased))

                        Just maxBound ->
                            if current >= maxBound then
                                Nothing

                            else
                                Just (msgTag (String.fromInt <| min increased maxBound))
            in
            Element.row [ Element.Border.solid, Element.Border.width 1, Element.Border.rounded 4 ]
                [ numberSideButton decrementMsg "−"
                , textField
                , numberSideButton incrementMsg "+"
                ]


numberSideButton : Maybe msg -> String -> Element msg
numberSideButton maybeMsg label =
    let
        textColor =
            if maybeMsg == Nothing then
                Style.lightGrey

            else
                Style.black
    in
    Element.Input.button
        [ height fill
        , width (Element.px 44)
        , Element.Font.center
        , Element.Font.color textColor
        ]
        { onPress = maybeMsg, label = Element.text label }



-- Float input


displayFloatErrors : Result (List NumberInput.FloatError) a -> Element msg
displayFloatErrors result =
    case result of
        Ok _ ->
            Element.none

        Err errors ->
            List.map Debug.toString errors
                |> String.join ", "
                |> Element.text


floatInput : NumberInput.Field Float NumberInput.FloatError -> (String -> msg) -> String -> Element msg
floatInput field msgTag label =
    let
        textField =
            Element.Input.text [ Element.Border.width 0, Element.Font.center, width (Element.px 140) ]
                { onChange = msgTag
                , text = field.input
                , placeholder = Nothing
                , label = Element.Input.labelHidden label
                }
    in
    case field.decodedInput of
        Err _ ->
            Element.row [ Element.Border.solid, Element.Border.width 1, Element.Border.rounded 4 ]
                [ numberSideButton Nothing "−"
                , textField
                , numberSideButton Nothing "+"
                ]

        Ok current ->
            let
                increased =
                    field.increase current

                decreased =
                    field.decrease current

                decrementMsg =
                    case field.min of
                        Nothing ->
                            Just (msgTag (String.fromFloat decreased))

                        Just minBound ->
                            if current <= minBound then
                                Nothing

                            else
                                Just (msgTag (String.fromFloat <| max decreased minBound))

                incrementMsg =
                    case field.max of
                        Nothing ->
                            Just (msgTag (String.fromFloat increased))

                        Just maxBound ->
                            if current >= maxBound then
                                Nothing

                            else
                                Just (msgTag (String.fromFloat <| min increased maxBound))
            in
            Element.row [ Element.Border.solid, Element.Border.width 1, Element.Border.rounded 4 ]
                [ numberSideButton decrementMsg "−"
                , textField
                , numberSideButton incrementMsg "+"
                ]



-- toggle


toggle : (Bool -> Msg) -> Bool -> Float -> String -> Element Msg
toggle msg checked toggleHeight label =
    Element.Input.checkbox [] <|
        { onChange = msg
        , label = Element.Input.labelHidden label
        , checked = checked
        , icon =
            toggleCheckboxWidget
                { offColor = Style.lightGrey
                , onColor = Style.green
                , sliderColor = Style.white
                , toggleWidth = 2 * round toggleHeight
                , toggleHeight = round toggleHeight
                }
        }


toggleCheckboxWidget : { offColor : Element.Color, onColor : Element.Color, sliderColor : Element.Color, toggleWidth : Int, toggleHeight : Int } -> Bool -> Element msg
toggleCheckboxWidget { offColor, onColor, sliderColor, toggleWidth, toggleHeight } checked =
    let
        pad =
            3

        sliderSize =
            toggleHeight - 2 * pad

        translation =
            (toggleWidth - sliderSize - pad)
                |> String.fromInt
    in
    Element.el
        [ Element.Background.color <|
            if checked then
                onColor

            else
                offColor
        , Element.width <| Element.px <| toggleWidth
        , Element.height <| Element.px <| toggleHeight
        , Element.Border.rounded (toggleHeight // 2)
        , Element.inFront <|
            Element.el [ Element.height Element.fill ] <|
                Element.el
                    [ Element.Background.color sliderColor
                    , Element.Border.rounded <| sliderSize // 2
                    , Element.width <| Element.px <| sliderSize
                    , Element.height <| Element.px <| sliderSize
                    , Element.centerY
                    , Element.moveRight pad
                    , Element.htmlAttribute <|
                        Html.Attributes.style "transition" ".4s"
                    , Element.htmlAttribute <|
                        if checked then
                            Html.Attributes.style "transform" <| "translateX(" ++ translation ++ "px)"

                        else
                            Html.Attributes.class ""
                    ]
                    (Element.text "")
        ]
        (Element.text "")


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
